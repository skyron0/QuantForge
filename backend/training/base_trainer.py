from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from backend.training.models import TrainingConfig


class BaseTrainer(ABC):
    """Abstract base for all ML trainers."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfig,
    ) -> Any:
        """Train and return the model object."""
        pass

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        pass

    @abstractmethod
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray | None:
        """Return probability predictions (classification only). None for regression."""
        pass

    @abstractmethod
    def get_feature_importance(
        self, model: Any, feature_names: list[str]
    ) -> Dict[str, float]:
        """Return feature name → importance mapping."""
        pass


class TrainerRegistry:
    """Maps model_type strings to BaseTrainer instances."""

    _trainers: Dict[str, BaseTrainer] = {}

    @classmethod
    def register(cls, model_type: str, trainer: BaseTrainer) -> None:
        cls._trainers[model_type] = trainer

    @classmethod
    def get(cls, model_type: str) -> BaseTrainer:
        if model_type not in cls._trainers:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(cls._trainers.keys())}"
            )
        return cls._trainers[model_type]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._trainers.keys())

    @classmethod
    def clear(cls) -> None:
        cls._trainers.clear()
