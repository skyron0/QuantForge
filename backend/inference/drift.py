import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Any, List, Optional, Callable

from backend.training.registry import ModelRegistry
from backend.inference.drift_detector import FeatureDriftDetector, DriftReport

logger = logging.getLogger(__name__)


class DriftObserver(ABC):
    """
    Observer interface for recording feature observations to monitor data drift.
    """

    @abstractmethod
    def on_observation(self, model_version: str, features: Dict[str, float]) -> None:
        """
        Invoked when a successful inference request occurs.
        """
        pass


class ActiveDriftObserver(DriftObserver):
    """
    Concrete drift observer that tracks runtime feature observations in bounded rolling windows
    and triggers distribution drift checks.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        detector: Optional[FeatureDriftDetector] = None,
        window_size: int = 1000,
        min_samples: int = 100,
    ):
        self.registry = registry or ModelRegistry()
        self.detector = detector or FeatureDriftDetector()
        self.window_size = window_size
        self.min_samples = min_samples
        self._observations: Dict[str, Dict[str, deque]] = {}
        self._lock = threading.Lock()
        self.callbacks: List[Callable[[DriftReport], None]] = []

    def on_observation(self, model_version: str, features: Dict[str, float]) -> None:
        """
        Adds new runtime feature observations to rolling queue. If minimum samples count
        is reached, triggers metric calculation.
        """
        if not features:
            return

        with self._lock:
            if model_version not in self._observations:
                self._observations[model_version] = {
                    feat: deque(maxlen=self.window_size) for feat in features
                }

            # Enqueue observations to prevent memory leaks (bounded window maxlen)
            for feat, val in features.items():
                if feat in self._observations[model_version]:
                    self._observations[model_version][feat].append(val)

            # Check sample size of first feature
            first_feat = next(iter(features))
            size = len(self._observations[model_version][first_feat])

        # Evaluate drift if sample count reaches threshold
        if size >= self.min_samples:
            self.evaluate_and_report(model_version)

    def evaluate_and_report(self, model_version: str) -> Optional[DriftReport]:
        """
        Pulls model registration record, builds DriftReport, and triggers callback layers.
        """
        try:
            model_record = self.registry.repo.get(model_version)
        except Exception as e:
            logger.error(f"Registry lookup failed for model {model_version} drift check: {str(e)}")
            return None

        if not model_record:
            logger.warning(f"Registered model {model_version} not found for drift check.")
            return None

        drift_baseline = getattr(model_record, "drift_baseline", None)
        if not drift_baseline:
            logger.warning(f"Model version {model_version} has no registered drift baseline.")
            return None

        # Thread-safe snapshot copy of deques to list
        with self._lock:
            if model_version not in self._observations:
                return None
            snapshot = {feat: list(dq) for feat, dq in self._observations[model_version].items()}

        report = self.detector.generate_report(model_version, drift_baseline, snapshot)

        # Trigger monitoring / telemetry hooks
        for cb in self.callbacks:
            try:
                cb(report)
            except Exception as e:
                logger.error(f"Error executing drift callback: {str(e)}")

        return report

