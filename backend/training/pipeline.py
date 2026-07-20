import os
import time
import random
import uuid
import datetime
from typing import Optional

import numpy as np
import pandas as pd

from backend.training.models import (
    TrainingConfig,
    TrainingResult,
    TrainingMetrics,
    ModelArtifact,
)
from backend.training.base_trainer import TrainerRegistry
from backend.training.validator import DatasetValidator
from backend.training.evaluator import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from backend.training.prediction_model import PredictionModel
from backend.training.persistence import (
    save_model,
    generate_manifest,
    save_manifest,
    determine_candidate_status,
)
from backend.training.registry import ModelRegistry


class TrainingPipeline:

    def __init__(
        self,
        output_dir: str = "data/models",
        experiment_repo=None,
    ):
        self.output_dir = output_dir
        self.validator = DatasetValidator()
        self.experiment_repo = experiment_repo

    def run(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        dataset_version: str = "",
    ) -> TrainingResult:
        # 1. Seed everything
        self._seed_all(config.random_seed)

        # 2. Auto-detect feature columns if not specified
        if not config.feature_columns:
            exclude = {"symbol", "timestamp", "label", "split"}
            config.feature_columns = [
                c for c in df.columns if c not in exclude
            ]

        # 3. Validate
        self.validator.validate(df, config.feature_columns, config.label_column)

        # 4. Split
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "val"]
        df_test = df[df["split"] == "test"] if "test" in df["split"].values else None

        X_train = df_train[config.feature_columns].values
        y_train = df_train[config.label_column].values
        X_val = df_val[config.feature_columns].values
        y_val = df_val[config.label_column].values

        # 5. Train
        trainer = TrainerRegistry.get(config.model_type)
        start = time.time()
        model = trainer.train(X_train, y_train, X_val, y_val, config)
        duration = time.time() - start

        # 6. Evaluate
        y_train_pred = trainer.predict(model, X_train)
        y_val_pred = trainer.predict(model, X_val)
        y_val_proba = trainer.predict_proba(model, X_val)

        if config.task_type == "classification":
            y_train_proba = trainer.predict_proba(model, X_train)
            train_metrics = compute_classification_metrics(
                y_train, y_train_pred, y_train_proba
            )
            val_metrics = compute_classification_metrics(
                y_val, y_val_pred, y_val_proba
            )
        else:
            train_metrics = compute_regression_metrics(y_train, y_train_pred)
            val_metrics = compute_regression_metrics(y_val, y_val_pred)

        test_metrics = None
        if df_test is not None and len(df_test) > 0:
            X_test = df_test[config.feature_columns].values
            y_test = df_test[config.label_column].values
            y_test_pred = trainer.predict(model, X_test)
            if config.task_type == "classification":
                y_test_proba = trainer.predict_proba(model, X_test)
                test_metrics = compute_classification_metrics(
                    y_test, y_test_pred, y_test_proba
                )
            else:
                test_metrics = compute_regression_metrics(y_test, y_test_pred)

        # 7. Feature importance
        importances = trainer.get_feature_importance(
            model, config.feature_columns
        )

        # 8. Build result
        result = TrainingResult(
            model=model,
            model_type=config.model_type,
            task_type=config.task_type,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importances=importances,
            training_duration_seconds=duration,
            config=config,
        )

        # 9. Candidate selection
        result.candidate_status = determine_candidate_status(
            result, config.candidate_thresholds
        )

        # 10. Persist
        model_version = str(uuid.uuid4())
        version_dir = os.path.join(self.output_dir, model_version)
        os.makedirs(version_dir, exist_ok=True)

        model_path = os.path.join(version_dir, "model.joblib")
        save_model(model, model_path)

        manifest = generate_manifest(
            config=config,
            result=result,
            model_version=model_version,
            model_path=model_path,
            dataset_version=dataset_version,
        )
        manifest_path = os.path.join(version_dir, "manifest.json")
        save_manifest(manifest, manifest_path)

        # 10.5 Register in model registry
        registry = ModelRegistry(experiment_repo=self.experiment_repo)
        registry.register(
            model_version=model_version,
            dataset_version=dataset_version,
            model_path=model_path,
            manifest=manifest,
        )

        # 11. Experiment tracking
        if self.experiment_repo is not None:
            self._track_experiment(config, result, manifest, model_version)

        return result

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def _track_experiment(
        self,
        config: TrainingConfig,
        result: TrainingResult,
        manifest: dict,
        model_version: str,
    ) -> None:
        from backend.experiment.models import (
            Experiment,
            ExperimentRun,
            ExperimentArtifact,
            ExperimentSummary,
        )
        from backend.experiment.git_helper import get_git_commit_hash

        exp_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        run = ExperimentRun(
            parameter_set=config.hyperparameters,
            metrics=result.val_metrics.to_dict(),
            artifacts=[
                ExperimentArtifact(
                    name="trained_model",
                    artifact_type="model",
                    path=manifest.get("model_path", ""),
                    content_summary=f"{config.model_type} {config.task_type} model",
                )
            ],
        )

        summary = ExperimentSummary(
            total_runs=1,
            best_run_parameters=config.hyperparameters,
            best_run_metrics=result.val_metrics.to_dict(),
            aggregated_metrics={
                "feature_importance": result.feature_importances,
                "candidate_status": result.candidate_status,
            },
        )

        experiment = Experiment(
            experiment_id=exp_id,
            experiment_name=f"Training_{config.model_type}_{timestamp.replace(':', '-')}",
            timestamp=timestamp,
            git_commit_hash=get_git_commit_hash(),
            strategy_name=config.model_type,
            strategy_parameters=config.hyperparameters,
            optimization_parameters={"random_seed": config.random_seed},
            dataset_identifier=manifest.get("dataset_version", ""),
            timeframe="",
            symbols=[],
            training_window=None,
            testing_window=None,
            walk_forward_config={},
            performance_metrics=result.val_metrics.to_dict(),
            notes=f"Candidate: {result.candidate_status}",
            runs=[run],
            summary=summary,
        )

        if self.experiment_repo is not None:
            self.experiment_repo.save(experiment)
