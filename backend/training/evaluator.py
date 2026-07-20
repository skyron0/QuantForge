import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from backend.training.models import TrainingMetrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> TrainingMetrics:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    if y_proba is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2:
                # Binary: use probability of the positive class
                if y_proba.ndim == 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba[:, 1])
                    )
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba)
                    )
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                )
        except (ValueError, IndexError):
            metrics["roc_auc"] = 0.0

    return TrainingMetrics(metrics=metrics)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> TrainingMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return TrainingMetrics(
        metrics={
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)),
        }
    )
