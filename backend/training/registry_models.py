from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import datetime

from backend.training.lifecycle import ModelStatus


@dataclass
class TransitionEvent:
    from_status: str
    to_status: str
    timestamp: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_status": self.from_status,
            "to_status": self.to_status,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransitionEvent":
        return cls(
            from_status=d["from_status"],
            to_status=d["to_status"],
            timestamp=d["timestamp"],
            notes=d.get("notes"),
        )


@dataclass
class RegisteredModel:
    model_version: str
    dataset_version: str
    feature_version: str
    label_version: str
    experiment_id: str
    git_commit: Optional[str]
    trainer: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    creation_timestamp: str
    model_path: str
    status: ModelStatus = ModelStatus.TRAINING
    transition_history: List[TransitionEvent] = field(default_factory=list)
    approval_notes: Optional[str] = None
    walk_forward_metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_sha256: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    drift_baseline: Dict[str, Any] = field(default_factory=dict)
    calibration_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "trainer": self.trainer,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "creation_timestamp": self.creation_timestamp,
            "model_path": self.model_path,
            "status": self.status.value,
            "transition_history": [e.to_dict() for e in self.transition_history],
            "approval_notes": self.approval_notes,
            "walk_forward_metrics": self.walk_forward_metrics,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "drift_baseline": self.drift_baseline,
            "calibration_metadata": self.calibration_metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegisteredModel":
        history = [TransitionEvent.from_dict(h) for h in d.get("transition_history", [])]
        return cls(
            model_version=d["model_version"],
            dataset_version=d["dataset_version"],
            feature_version=d["feature_version"],
            label_version=d["label_version"],
            experiment_id=d["experiment_id"],
            git_commit=d.get("git_commit"),
            trainer=d["trainer"],
            hyperparameters=d["hyperparameters"],
            metrics=d["metrics"],
            feature_importance=d.get("feature_importance", {}),
            creation_timestamp=d["creation_timestamp"],
            model_path=d["model_path"],
            status=ModelStatus(d["status"]),
            transition_history=history,
            approval_notes=d.get("approval_notes"),
            walk_forward_metrics=d.get("walk_forward_metrics", {}),
            artifact_sha256=d.get("artifact_sha256"),
            artifact_size_bytes=d.get("artifact_size_bytes"),
            drift_baseline=d.get("drift_baseline", {}),
            calibration_metadata=d.get("calibration_metadata", {}),
        )
