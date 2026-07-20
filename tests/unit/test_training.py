import pytest
import os
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

import backend.training.trainers
from backend.training.models import TrainingConfig, TrainingResult, TrainingMetrics
from backend.training.base_trainer import TrainerRegistry, BaseTrainer
from backend.training.validator import DatasetValidator, DatasetValidationError
from backend.training.evaluator import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from backend.training.prediction_model import PredictionModel
from backend.training.persistence import (
    save_model,
    load_model,
    generate_manifest,
    determine_candidate_status,
)
from backend.training.pipeline import TrainingPipeline


# ────────────── 1. Registry Tests ──────────────

def test_trainer_registry():
    """Verify registry registration and retrieval."""
    class DummyTrainer(BaseTrainer):
        def train(self, X_train, y_train, X_val, y_val, config):
            return "dummy_model"
        def predict(self, model, X):
            return np.zeros(len(X))
        def predict_proba(self, model, X):
            return None
        def get_feature_importance(self, model, feature_names):
            return {name: 1.0 for name in feature_names}

    TrainerRegistry.register("dummy", DummyTrainer())
    assert "dummy" in TrainerRegistry.available()

    trainer = TrainerRegistry.get("dummy")
    assert isinstance(trainer, DummyTrainer)

    with pytest.raises(ValueError, match="Unknown model type"):
        TrainerRegistry.get("invalid_trainer")


# ────────────── 2. Validator Tests ──────────────

def test_dataset_validator():
    validator = DatasetValidator(imbalance_threshold=0.05)
    feature_cols = ["f1", "f2"]
    label_col = "label"

    # Empty dataset
    df_empty = pd.DataFrame()
    with pytest.raises(DatasetValidationError, match="Dataset is empty"):
        validator.validate(df_empty, feature_cols, label_col)

    # Missing columns
    df_missing = pd.DataFrame({"f1": [1.0, 2.0]})
    with pytest.raises(DatasetValidationError, match="Missing columns"):
        validator.validate(df_missing, feature_cols, label_col)

    # NaN values
    df_nan = pd.DataFrame({
        "f1": [1.0, np.nan, 3.0],
        "f2": [2.0, 4.0, 6.0],
        "label": [1, 0, 1]
    })
    with pytest.raises(DatasetValidationError, match="NaN values found"):
        validator.validate(df_nan, feature_cols, label_col)

    # Infinite values
    df_inf = pd.DataFrame({
        "f1": [1.0, np.inf, 3.0],
        "f2": [2.0, 4.0, 6.0],
        "label": [1, 0, 1]
    })
    with pytest.raises(DatasetValidationError, match="Infinite values found"):
        validator.validate(df_inf, feature_cols, label_col)

    # Constant columns
    df_const = pd.DataFrame({
        "f1": [2.0, 2.0, 2.0],
        "f2": [1.0, 2.0, 3.0],
        "label": [1, 0, 1]
    })
    with pytest.raises(DatasetValidationError, match="Constant columns"):
        validator.validate(df_const, feature_cols, label_col)

    # Single class target
    df_single_class = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [2.0, 4.0, 5.0],
        "label": [1, 1, 1]
    })
    with pytest.raises(DatasetValidationError, match="Single-class target"):
        validator.validate(df_single_class, feature_cols, label_col)

    # Severe class imbalance (imbalance check requires enough rows to hit ratio)
    # Target label 1: 19 rows, label 0: 1 row = 5% ratio (not balanced, trigger <=0.05 limit depending on strict balance)
    df_imbalanced = pd.DataFrame({
        "f1": list(range(20)),
        "f2": list(range(20)),
        "label": [1]*19 + [0]
    })
    # Since ratio is 0.05, we set threshold to 0.10 to trigger it
    validator_strict = DatasetValidator(imbalance_threshold=0.10)
    with pytest.raises(DatasetValidationError, match="Severe class imbalance"):
        validator_strict.validate(df_imbalanced, feature_cols, label_col)


# ────────────── 3. Metrics Tests ──────────────

def test_evaluation_metrics():
    # Classification
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    y_proba = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.9, 0.1]
    ])
    cls_metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert cls_metrics.get("accuracy") == 0.8
    assert "precision" in cls_metrics.to_dict()
    assert "recall" in cls_metrics.to_dict()
    assert "f1" in cls_metrics.to_dict()
    assert cls_metrics.get("roc_auc") > 0.0

    # Regression
    y_true_reg = np.array([10.0, 20.0, 30.0])
    y_pred_reg = np.array([11.0, 19.0, 32.0])
    reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
    assert reg_metrics.get("mae") == pytest.approx(1.333, abs=0.01)
    assert reg_metrics.get("rmse") == pytest.approx(1.414, abs=0.01)
    assert reg_metrics.get("r2") > 0.0


# ────────────── 4. Persistence & Candidate Selection ──────────────

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_model_persistence(temp_dir):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # Simple fitting
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    y = np.array([1, 0, 1])
    model.fit(X, y)

    model_path = os.path.join(temp_dir, "model.joblib")
    save_model(model, model_path)
    assert os.path.isfile(model_path)

    loaded = load_model(model_path)
    assert isinstance(loaded, LogisticRegression)
    assert np.allclose(loaded.predict(X), model.predict(X))


def test_candidate_selection():
    metrics = TrainingMetrics(metrics={"accuracy": 0.60, "f1": 0.58})
    result = TrainingResult(
        model=None,
        model_type="lightgbm",
        task_type="classification",
        train_metrics=metrics,
        val_metrics=metrics
    )

    # Threshold met
    status = determine_candidate_status(result, {"accuracy": 0.55})
    assert status == "YES"

    # Threshold missed
    status_fail = determine_candidate_status(result, {"accuracy": 0.65})
    assert status_fail == "NO"


# ────────────── 5. Determinism & Pipeline ──────────────

def _get_balanced_df(n_samples=100) -> pd.DataFrame:
    np.random.seed(42)
    f1 = np.random.randn(n_samples)
    f2 = np.random.randn(n_samples)
    noise = np.random.randn(n_samples) * 0.1
    # Simple linear decision path
    y = (f1 + f2 + noise > 0).astype(int)

    splits = ["train"] * int(n_samples * 0.6) + ["val"] * int(n_samples * 0.2) + ["test"] * (n_samples - int(n_samples * 0.6) - int(n_samples * 0.2))

    return pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "label": y,
        "split": splits,
        "symbol": "BTCUSD"
    })


def test_deterministic_training():
    """Verify that same seed generates perfectly identical results across runs."""
    df = _get_balanced_df()
    feature_cols = ["f1", "f2"]

    config1 = TrainingConfig(
        model_type="lightgbm",
        task_type="classification",
        hyperparameters={"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1},
        random_seed=42,
        feature_columns=feature_cols
    )
    config2 = TrainingConfig(
        model_type="lightgbm",
        task_type="classification",
        hyperparameters={"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1},
        random_seed=42,
        feature_columns=feature_cols
    )

    pipeline = TrainingPipeline()
    res1 = pipeline.run(df, config1)
    res2 = pipeline.run(df, config2)

    # Verify identical validation metrics
    assert res1.val_metrics.to_dict() == res2.val_metrics.to_dict()
    assert res1.feature_importances == res2.feature_importances


def test_pipeline_experiment_integration(temp_dir):
    """Verify training run saves models, generates manifest, and integrates with experiment tracking."""
    class FakeExperimentRepo:
        def __init__(self):
            self.saved = []
        def save(self, experiment):
            self.saved.append(experiment)

    df = _get_balanced_df()
    config = TrainingConfig(
        model_type="xgboost",
        task_type="classification",
        hyperparameters={"n_estimators": 5, "max_depth": 2},
        random_seed=42,
        feature_columns=["f1", "f2"],
        candidate_thresholds={"accuracy": 0.50}
    )

    repo = FakeExperimentRepo()
    pipeline = TrainingPipeline(output_dir=temp_dir, experiment_repo=repo)
    result = pipeline.run(df, config, dataset_version="ds-999")

    # Assertions: 2 experiments saved (1 from registry register NONE->TRAINING, 1 from training pipeline)
    assert len(repo.saved) == 2
    
    exp_reg = repo.saved[0]
    assert "Lifecycle" in exp_reg.experiment_name
    assert "NONE to TRAINING" in exp_reg.notes

    exp_train = repo.saved[1]
    assert exp_train.dataset_identifier == "ds-999"
    assert exp_train.summary is not None
    assert exp_train.summary.aggregated_metrics["candidate_status"] == "YES"

    # Manifest and model saved check
    folders = os.listdir(temp_dir)
    assert len(folders) == 1
    ver_dir = os.path.join(temp_dir, folders[0])
    assert os.path.isfile(os.path.join(ver_dir, "model.joblib"))
    assert os.path.isfile(os.path.join(ver_dir, "manifest.json"))

    with open(os.path.join(ver_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    assert manifest["candidate_status"] == "YES"
    assert manifest["dataset_version"] == "ds-999"
    assert "accuracy" in manifest["val_metrics"]
