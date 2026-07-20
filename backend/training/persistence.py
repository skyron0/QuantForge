import os
import json
import datetime
import joblib
from typing import Any, Dict, Optional

from backend.training.models import TrainingConfig, TrainingResult, ModelArtifact
from backend.experiment.git_helper import get_git_commit_hash


def save_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)


def generate_manifest(
    config: TrainingConfig,
    result: TrainingResult,
    model_version: str,
    model_path: str,
    dataset_version: str = "",
    feature_version: str = "v1",
    label_version: str = "v1",
    experiment_id: str = "",
) -> Dict[str, Any]:
    return {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "label_version": label_version,
        "experiment_id": experiment_id,
        "git_commit": get_git_commit_hash(),
        "trainer": config.model_type,
        "task_type": config.task_type,
        "hyperparameters": config.hyperparameters,
        "random_seed": config.random_seed,
        "train_metrics": result.train_metrics.to_dict(),
        "val_metrics": result.val_metrics.to_dict(),
        "test_metrics": result.test_metrics.to_dict() if result.test_metrics else {},
        "feature_importance": result.feature_importances,
        "candidate_status": result.candidate_status,
        "training_duration_seconds": result.training_duration_seconds,
        "model_path": model_path,
        "creation_timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }


def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)


def determine_candidate_status(
    result: TrainingResult,
    thresholds: Dict[str, float],
) -> str:
    """Return 'YES' if all threshold conditions are met, 'NO' otherwise."""
    if not thresholds:
        return "NO"

    val_metrics = result.val_metrics.to_dict()
    for metric_name, threshold_value in thresholds.items():
        actual = val_metrics.get(metric_name, 0.0)
        if actual < threshold_value:
            return "NO"

    return "YES"
