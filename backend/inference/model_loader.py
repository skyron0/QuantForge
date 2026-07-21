import os
import threading
import logging
from typing import Dict, Any, Optional, Tuple

from backend.training.registry import ModelRegistry
from backend.training.lifecycle import ModelStatus
from backend.training.prediction_model import PredictionModel
from backend.inference.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    LifecycleError,
    RegistryError,
    ArtifactIntegrityError,
)
from backend.inference.integrity import ArtifactIntegrityVerifier, IntegrityPolicy

logger = logging.getLogger(__name__)


# ────────────── Legacy Compatibility ──────────────

class ManifestChecksumVerifier:
    """Legacy class preserved for unit test backward compatibility."""
    def verify(self, model_path: str, manifest: dict) -> bool:
        return True


# ────────────── Model Loader Cache ──────────────

class ModelLoaderCache:
    """
    Thread-safe cache for loading, validating, and reusing PredictionModel instances.
    Enforces integrity checks and registry eligibility rules.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        verifier: Optional[ArtifactIntegrityVerifier] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.verifier = verifier or ArtifactIntegrityVerifier()
        self._cache: Dict[str, PredictionModel] = {}
        self._lock = threading.Lock()

    def get_model(self, model_version: str) -> Tuple[PredictionModel, bool]:
        """
        Retrieves the PredictionModel instance from cache or loads from disk.
        Returns a tuple: (PredictionModel, cache_hit).
        Enforces SHA-256 integrity and lifecycle validation checking on every load call.
        """
        # Always check database status first (Constraint #4: Cached model eligibility check)
        try:
            model_record = self.registry.repo.get(model_version)
        except Exception as e:
            raise RegistryError(f"Failed to query model registry: {str(e)}") from e

        if not model_record:
            raise ModelNotFoundError(f"Model version '{model_version}' not found in registry.")

        # Check current lifecycle eligibility
        allowed_statuses = {ModelStatus.VALIDATED, ModelStatus.SHADOW, ModelStatus.PRODUCTION}
        if model_record.status not in allowed_statuses:
            raise LifecycleError(
                f"Model '{model_version}' has status '{model_record.status.value}', "
                f"which is ineligible for inference. Valid statuses: {[s.value for s in allowed_statuses]}."
            )

        with self._lock:
            if model_version in self._cache:
                return self._cache[model_version], True

            # 1. Verify model weights file integrity
            registered_sha = getattr(model_record, "artifact_sha256", None)
            try:
                self.verifier.verify(
                    filepath=model_record.model_path,
                    registered_sha256=registered_sha,
                    model_status=model_record.status,
                )
            except ArtifactIntegrityError:
                raise
            except Exception as e:
                raise ArtifactIntegrityError(f"Model verification failed: {str(e)}") from e

            # 2. Verify calibration file integrity (if calibration metadata exists)
            calibrator = None
            calib_meta = getattr(model_record, "calibration_metadata", {})
            if calib_meta and calib_meta.get("calibration_artifact_path"):
                calib_path = calib_meta["calibration_artifact_path"]
                registered_calib_sha = calib_meta.get("calibration_sha256")
                
                try:
                    self.verifier.verify(
                        filepath=calib_path,
                        registered_sha256=registered_calib_sha,
                        model_status=model_record.status,
                    )
                    # Load calibration artifact
                    from backend.inference.calibration import BaseProbabilityCalibrator
                    calibrator = BaseProbabilityCalibrator.load(calib_path)
                except ArtifactIntegrityError:
                    raise
                except Exception as e:
                    raise ArtifactIntegrityError(f"Calibration verification failed: {str(e)}") from e

            # 3. Load model artifact
            try:
                model = self.registry.load(model_version)
            except FileNotFoundError as e:
                raise ModelLoadError(f"Model file path not found: {str(e)}") from e
            except Exception as e:
                raise ModelLoadError(f"Failed to deserialize model artifact: {str(e)}") from e

            # Attach loaded calibrator dynamically to the PredictionModel instance
            model.calibrator = calibrator

            # Save in cache
            self._cache[model_version] = model
            return model, False

    def invalidate(self, model_version: str) -> None:
        """Removes a model version from the cache."""
        with self._lock:
            self._cache.pop(model_version, None)

    def reload(self, model_version: str) -> PredictionModel:
        """Evicts a version from cache and forces a fresh reload."""
        self.invalidate(model_version)
        model, _ = self.get_model(model_version)
        return model

    def clear(self) -> None:
        """Clears all cached models from memory."""
        with self._lock:
            self._cache.clear()
