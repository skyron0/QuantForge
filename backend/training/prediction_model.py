import numpy as np
from typing import Dict, Any


class PredictionModel:
    """
    Provider-independent prediction abstraction.
    Downstream QuantForge code (strategies, decision fusion) imports only this class.
    Never imports LightGBM, XGBoost, CatBoost directly.
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        task_type: str,
        feature_columns: list[str],
        model_version: str = "",
    ):
        self._model = model
        self.model_type = model_type
        self.task_type = task_type
        self.feature_columns = feature_columns
        self.model_version = model_version
        self.calibrator: Any = None

    def predict(self, features: Dict[str, float]) -> float:
        """Predict from a feature dict. Returns single prediction value."""
        X = np.array([[features[c] for c in self.feature_columns]])
        pred = self._model.predict(X)
        return float(pred[0])

    def predict_proba(self, features: Dict[str, float]) -> np.ndarray | None:
        """Return class probabilities (classification only)."""
        raw = self.raw_model
        if raw is self or not hasattr(raw, "predict_proba"):
            return None
        X = np.array([[features[c] for c in self.feature_columns]])
        res = raw.predict_proba(X)
        return res[0] if res is not None else None

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict from a 2D numpy array."""
        return self.raw_model.predict(X)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities for a 2D numpy array (classification only)."""
        raw = self.raw_model
        if raw is self or not hasattr(raw, "predict_proba"):
            return None
        return raw.predict_proba(X)

    @property
    def raw_model(self) -> Any:
        """Access underlying model (for advanced use only)."""
        return self._model
