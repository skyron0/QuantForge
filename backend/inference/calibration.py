import os
import joblib
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from backend.inference.exceptions import ArtifactIntegrityError


class BaseProbabilityCalibrator(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseProbabilityCalibrator":
        """Fit the calibrator using model predicted probabilities and validation labels."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the calibration transformation to raw probabilities, returning shape (N, 2)."""
        pass

    def save(self, filepath: str) -> None:
        """Persist the calibrator using joblib."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaseProbabilityCalibrator":
        """Load the calibrator artifact from disk."""
        if not os.path.exists(filepath):
            raise ArtifactIntegrityError(f"Calibration artifact '{filepath}' does not exist.")
        try:
            return joblib.load(filepath)
        except Exception as e:
            raise ArtifactIntegrityError(f"Failed to load calibration artifact: {str(e)}") from e


class PlattCalibrator(BaseProbabilityCalibrator):
    def __init__(self):
        self.lr_ = LogisticRegression(C=1e5, solver="liblinear")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        probs = self._extract_positive_class_probs(X)
        self.lr_.fit(probs.reshape(-1, 1), y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        probs = self._extract_positive_class_probs(X)
        calibrated_probs_pos = self.lr_.predict_proba(probs.reshape(-1, 1))[:, 1]
        return np.column_stack((1.0 - calibrated_probs_pos, calibrated_probs_pos))

    def _extract_positive_class_probs(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            if X.shape[1] == 2:
                # Column 1 represents class 1 (positive)
                return X[:, 1]
            return X[:, 0]
        return X


class IsotonicCalibrator(BaseProbabilityCalibrator):
    def __init__(self):
        self.ir_ = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        probs = self._extract_positive_class_probs(X)
        self.ir_.fit(probs, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        probs = self._extract_positive_class_probs(X)
        calibrated_probs_pos = self.ir_.predict(probs)
        return np.column_stack((1.0 - calibrated_probs_pos, calibrated_probs_pos))

    def _extract_positive_class_probs(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            if X.shape[1] == 2:
                return X[:, 1]
            return X[:, 0]
        return X


def calculate_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) for binary classification.
    """
    y = np.asarray(y)
    probs = np.asarray(probs)
    
    # Extract prediction probability of positive class
    if probs.ndim == 2:
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        else:
            probs = probs[:, 0]

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (probs == bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_calibration_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Computes Brier Score, Log Loss, and Expected Calibration Error (ECE).
    """
    pos_probs = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] == 2) else probs
    
    brier = float(brier_score_loss(y, pos_probs))
    loss = float(log_loss(y, probs))
    ece = calculate_ece(pos_probs, y)
    
    return {
        "brier_score": brier,
        "log_loss": loss,
        "ece": ece,
    }
