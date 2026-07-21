import datetime
from typing import Dict, Any, List, Optional
import uuid

from backend.training.lifecycle import ModelStatus, validate_transition
from backend.training.registry_models import RegisteredModel, TransitionEvent
from backend.training.registry_repo import LocalModelRegistryRepository
from backend.training.prediction_model import PredictionModel
from backend.training.persistence import load_model


class ModelRegistry:
    """
    State orchestrator for managing machine learning model versions,
    lifecycle state transitions, performance rankings, and persistence.
    """

    def __init__(self, repository: Optional[LocalModelRegistryRepository] = None, experiment_repo=None):
        self.repo = repository or LocalModelRegistryRepository()
        self.experiment_repo = experiment_repo

    def register(
        self,
        model_version: str,
        dataset_version: str,
        model_path: str,
        manifest: Dict[str, Any],
    ) -> RegisteredModel:
        """Register a newly trained model in the TRAINING state."""
        # Check if already registered
        existing = self.repo.get(model_version)
        if existing:
            return existing

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        initial_event = TransitionEvent(
            from_status="NONE",
            to_status="TRAINING",
            timestamp=now,
            notes="Initial training registration.",
        )

        model = RegisteredModel(
            model_version=model_version,
            dataset_version=dataset_version,
            feature_version=manifest.get("feature_version", "v1"),
            label_version=manifest.get("label_version", "v1"),
            experiment_id=manifest.get("experiment_id", ""),
            git_commit=manifest.get("git_commit"),
            trainer=manifest.get("trainer", ""),
            hyperparameters=manifest.get("hyperparameters", {}),
            metrics=manifest.get("val_metrics", {}),  # Use val_metrics standardly
            feature_importance=manifest.get("feature_importance", {}),
            creation_timestamp=manifest.get("creation_timestamp", now),
            model_path=model_path,
            status=ModelStatus.TRAINING,
            transition_history=[initial_event],
            approval_notes=manifest.get("notes"),
            artifact_sha256=manifest.get("artifact_sha256"),
            artifact_size_bytes=manifest.get("artifact_size_bytes"),
            drift_baseline=manifest.get("drift_baseline", {}),
            calibration_metadata=manifest.get("calibration_metadata", {}),
        )
        self.repo.save(model)
        self._log_experiment_transition(model, "NONE", "TRAINING")
        return model

    def load(self, model_version: str) -> PredictionModel:
        """Load and return PredictionModel provider-independent sarmali."""
        model_record = self.repo.get(model_version)
        if not model_record:
            raise FileNotFoundError(f"Model version '{model_version}' not found in registry.")

        raw = load_model(model_record.model_path)
        # Parse task type from manifest or default
        task_type = "classification"
        if "regression" in model_record.trainer or "mae" in model_record.metrics:
            task_type = "regression"

        feature_cols = list(model_record.feature_importance.keys())

        return PredictionModel(
            model=raw,
            model_type=model_record.trainer,
            task_type=task_type,
            feature_columns=feature_cols,
            model_version=model_version,
        )

    def list_models(self, status: Optional[str] = None) -> List[RegisteredModel]:
        models = self.repo.list_all()
        if status:
            target = ModelStatus(status)
            return [m for m in models if m.status == target]
        return models

    def find_candidates(self) -> List[RegisteredModel]:
        return self.list_models(status="CANDIDATE")

    def find_best(
        self,
        metric_name: str,
        task_type: str,
        status: Optional[str] = None,
    ) -> Optional[RegisteredModel]:
        """
        Locates the best performing model.
        Supports maximization (F1, Accuracy, R2, ROC-AUC) or minimization (MAE, RMSE).
        """
        models = self.list_models(status=status)
        # Filter models by trainer task type signature
        valid_models = []
        for m in models:
            is_reg = "mae" in m.metrics or "rmse" in m.metrics
            if task_type == "regression" and is_reg:
                valid_models.append(m)
            elif task_type == "classification" and not is_reg:
                valid_models.append(m)

        if not valid_models:
            return None

        # Determine metric orientation
        lower_is_better = metric_name.lower() in ("mae", "rmse", "mse")

        best_model = None
        best_value = float("inf") if lower_is_better else float("-inf")

        for m in valid_models:
            val = m.metrics.get(metric_name)
            if val is None:
                continue

            if lower_is_better:
                if val < best_value:
                    best_value = val
                    best_model = m
            else:
                if val > best_value:
                    best_value = val
                    best_model = m

        return best_model

    def promote(self, model_version: str, new_status: ModelStatus, notes: Optional[str] = None) -> None:
        """Promote model lifecycle sequence."""
        model = self.repo.get(model_version)
        if not model:
            raise FileNotFoundError(f"Model version '{model_version}' not found.")

        old_status = model.status
        validate_transition(old_status, new_status)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event = TransitionEvent(
            from_status=old_status.value,
            to_status=new_status.value,
            timestamp=now,
            notes=notes,
        )

        model.status = new_status
        model.transition_history.append(event)
        if notes:
            model.approval_notes = notes

        self.repo.save(model)
        self._log_experiment_transition(model, old_status.value, new_status.value, notes)

    def deprecate(self, model_version: str, notes: Optional[str] = None) -> None:
        self.promote(model_version, ModelStatus.DEPRECATED, notes)

    def archive(self, model_version: str, notes: Optional[str] = None) -> None:
        self.promote(model_version, ModelStatus.ARCHIVED, notes)

    def _log_experiment_transition(
        self,
        model: RegisteredModel,
        from_status: str,
        to_status: str,
        notes: Optional[str] = None,
    ) -> None:
        """Logs transitions into the Experiment Repository."""
        if self.experiment_repo is None:
            return

        from backend.experiment.models import (
            Experiment,
            ExperimentRun,
            ExperimentSummary,
        )
        from backend.experiment.git_helper import get_git_commit_hash

        exp_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        run = ExperimentRun(
            parameter_set={
                "model_version": model.model_version,
                "transition": f"{from_status}->{to_status}",
                "notes": notes or "",
            },
            metrics=model.metrics,
        )

        experiment = Experiment(
            experiment_id=exp_id,
            experiment_name=f"Lifecycle_{model.model_version[:8]}_{timestamp.replace(':', '-')}",
            timestamp=timestamp,
            git_commit_hash=get_git_commit_hash(),
            strategy_name=model.trainer,
            strategy_parameters=model.hyperparameters,
            optimization_parameters={},
            dataset_identifier=model.dataset_version,
            timeframe="",
            symbols=[],
            training_window=None,
            testing_window=None,
            walk_forward_config={},
            performance_metrics=model.metrics,
            notes=f"Lifecycle transition for model {model.model_version[:8]}: {from_status} to {to_status}. Notes: {notes or ''}",
            runs=[run],
            summary=ExperimentSummary(
                total_runs=1,
                best_run_parameters={},
                best_run_metrics=model.metrics,
                aggregated_metrics={"lifecycle_notes": notes or ""},
            ),
        )
        self.experiment_repo.save(experiment)
