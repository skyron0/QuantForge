from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import uuid


@dataclass
class TrainingConfig:
    model_type: str  # "lightgbm", "xgboost", "catboost"
    task_type: str  # "classification", "regression"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    feature_columns: List[str] = field(default_factory=list)
    label_column: str = "label"
    candidate_thresholds: Dict[str, float] = field(default_factory=dict)
    cv_splits: Optional[int] = None  # TimeSeriesSplit support (architecture-ready)


@dataclass
class TrainingMetrics:
    metrics: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.metrics.get(key, default)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.metrics)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "TrainingMetrics":
        return cls(metrics=dict(d))


@dataclass
class TrainingResult:
    model: Any  # The trained model object
    model_type: str
    task_type: str
    train_metrics: TrainingMetrics
    val_metrics: TrainingMetrics
    test_metrics: Optional[TrainingMetrics] = None
    feature_importances: Dict[str, float] = field(default_factory=dict)
    training_duration_seconds: float = 0.0
    candidate_status: str = "NO"  # "YES" or "NO"
    config: Optional[TrainingConfig] = None


@dataclass
class ModelArtifact:
    model_path: str
    manifest_path: str
    model_version: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""
