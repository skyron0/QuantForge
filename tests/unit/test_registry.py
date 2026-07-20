import os
import shutil
import tempfile
import pytest
import numpy as np

from backend.training.lifecycle import ModelStatus, InvalidStateTransitionError
from backend.training.registry_models import RegisteredModel, TransitionEvent
from backend.training.registry_repo import LocalModelRegistryRepository
from backend.training.registry import ModelRegistry
from backend.training.prediction_model import PredictionModel


@pytest.fixture
def temp_registry_db():
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "model_registry.json")
    repo = LocalModelRegistryRepository(db_path=db_path)
    yield repo
    shutil.rmtree(d)


# ────────────── 1. Lifecycle Transition Rules Tests ──────────────

def test_lifecycle_transitions():
    from backend.training.lifecycle import validate_transition

    # Identical is fine
    validate_transition(ModelStatus.TRAINING, ModelStatus.TRAINING)

    # Valid step-by-step
    validate_transition(ModelStatus.TRAINING, ModelStatus.CANDIDATE)
    validate_transition(ModelStatus.CANDIDATE, ModelStatus.VALIDATED)
    validate_transition(ModelStatus.VALIDATED, ModelStatus.SHADOW)
    validate_transition(ModelStatus.SHADOW, ModelStatus.PRODUCTION)

    # Valid out-of-band retirement transitions
    validate_transition(ModelStatus.TRAINING, ModelStatus.DEPRECATED)
    validate_transition(ModelStatus.SHADOW, ModelStatus.DEPRECATED)
    validate_transition(ModelStatus.PRODUCTION, ModelStatus.DEPRECATED)

    validate_transition(ModelStatus.CANDIDATE, ModelStatus.ARCHIVED)
    validate_transition(ModelStatus.DEPRECATED, ModelStatus.ARCHIVED)

    # Invalid sequence jumps
    with pytest.raises(InvalidStateTransitionError, match="Invalid lifecycle transition"):
        validate_transition(ModelStatus.TRAINING, ModelStatus.SHADOW)

    with pytest.raises(InvalidStateTransitionError, match="Invalid lifecycle transition"):
        validate_transition(ModelStatus.CANDIDATE, ModelStatus.PRODUCTION)

    # Cannot get out of ARCHIVED
    with pytest.raises(InvalidStateTransitionError, match="Cannot transition out of terminal state"):
        validate_transition(ModelStatus.ARCHIVED, ModelStatus.SHADOW)


# ────────────── 2. Registry Orchestrator Tests ──────────────

def test_registry_registration_and_promotion(temp_registry_db):
    registry = ModelRegistry(repository=temp_registry_db)

    manifest = {
        "feature_version": "v2",
        "label_version": "v1",
        "experiment_id": "exp-123",
        "git_commit": "abcdef",
        "trainer": "lightgbm",
        "val_metrics": {"accuracy": 0.85, "f1": 0.82},
        "feature_importance": {"rsi": 5.0, "ema": 12.0},
        "creation_timestamp": "2026-07-20T12:00:00Z",
    }

    # Register
    model = registry.register(
        model_version="ver-1.0",
        dataset_version="ds-100",
        model_path="dummy/path/model.joblib",
        manifest=manifest,
    )

    assert model.model_version == "ver-1.0"
    assert model.status == ModelStatus.TRAINING
    assert len(model.transition_history) == 1
    assert model.transition_history[0].to_status == "TRAINING"

    # Promote Training -> Candidate -> Validated
    registry.promote("ver-1.0", ModelStatus.CANDIDATE, notes="High accuracy")
    model_updated = temp_registry_db.get("ver-1.0")
    assert model_updated.status == ModelStatus.CANDIDATE
    assert len(model_updated.transition_history) == 2
    assert model_updated.transition_history[1].from_status == "TRAINING"
    assert model_updated.transition_history[1].to_status == "CANDIDATE"
    assert model_updated.approval_notes == "High accuracy"

    # Promotion to illegal status raises error
    with pytest.raises(InvalidStateTransitionError):
        registry.promote("ver-1.0", ModelStatus.PRODUCTION)

    # Deprecate
    registry.deprecate("ver-1.0", notes="Obsolete parameters")
    model_deprecated = temp_registry_db.get("ver-1.0")
    assert model_deprecated.status == ModelStatus.DEPRECATED

    # Archive
    registry.archive("ver-1.0", notes="Archiving model")
    model_archived = temp_registry_db.get("ver-1.0")
    assert model_archived.status == ModelStatus.ARCHIVED


class MockModelObject:
    def predict(self, X):
        return np.ones(len(X))


def test_registry_loading(temp_registry_db, tmp_path):
    # Save a fake joblib model
    import joblib
    model_obj = MockModelObject()
    model_file = tmp_path / "model.joblib"
    joblib.dump(model_obj, model_file)

    registry = ModelRegistry(repository=temp_registry_db)
    manifest = {
        "trainer": "xgboost",
        "val_metrics": {"accuracy": 0.90},
        "feature_importance": {"f1": 1.0, "f2": 2.0},
        "creation_timestamp": "2026-07-20T12:00:00Z"
    }

    registry.register(
        model_version="ver-large",
        dataset_version="ds-2",
        model_path=str(model_file),
        manifest=manifest
    )

    # Load sarmali PredictionModel
    pred_model = registry.load("ver-large")
    assert isinstance(pred_model, PredictionModel)
    assert pred_model.model_version == "ver-large"
    assert pred_model.model_type == "xgboost"
    assert pred_model.feature_columns == ["f1", "f2"]

    # Predict
    features = {"f1": 1.0, "f2": 3.4}
    res = pred_model.predict(features)
    assert res == 1.0


def test_registry_find_best(temp_registry_db):
    registry = ModelRegistry(repository=temp_registry_db)

    # Make three models
    m1_manifest = {
        "trainer": "lightgbm",
        "val_metrics": {"accuracy": 0.60, "f1": 0.58},
        "feature_importance": {"f1": 1.0},
        "creation_timestamp": "2026-07-20T12:00:00Z"
    }
    m2_manifest = {
        "trainer": "lightgbm",
        "val_metrics": {"accuracy": 0.80, "f1": 0.77},
        "feature_importance": {"f1": 1.0},
        "creation_timestamp": "2026-07-20T12:00:00Z"
    }
    m3_manifest = {
        "trainer": "lightgbm_regression", # Make trainer string indicate regression task
        "val_metrics": {"mae": 0.4},
        "feature_importance": {"f1": 1.0},
        "creation_timestamp": "2026-07-20T12:00:00Z"
    }

    registry.register("m1", "ds-1", "path1", m1_manifest)
    registry.register("m2", "ds-1", "path2", m2_manifest)
    registry.register("m3", "ds-1", "path3", m3_manifest)

    # All registered in TRAINING state. Promote m1, m2 to CANDIDATE
    registry.promote("m1", ModelStatus.CANDIDATE)
    registry.promote("m2", ModelStatus.CANDIDATE)
    registry.promote("m3", ModelStatus.CANDIDATE)

    # Find best classification model (maximization)
    best_cls = registry.find_best(metric_name="f1", task_type="classification", status="CANDIDATE")
    assert best_cls is not None
    assert best_cls.model_version == "m2"

    # Find best regression model (minimization)
    best_reg = registry.find_best(metric_name="mae", task_type="regression", status="CANDIDATE")
    assert best_reg is not None
    assert best_reg.model_version == "m3"


def test_registry_experiment_integration(temp_registry_db):
    class FakeExperimentRepository:
        def __init__(self):
            self.saved = []
        def save(self, exp):
            self.saved.append(exp)

    exp_repo = FakeExperimentRepository()
    registry = ModelRegistry(repository=temp_registry_db, experiment_repo=exp_repo)

    manifest = {
        "trainer": "catboost",
        "val_metrics": {"f1": 0.70},
        "feature_importance": {"f1": 1.0},
        "creation_timestamp": "2026-07-20T16:00:00Z"
    }

    registry.register("ver-cb", "ds-3", "path-cb", manifest)
    assert len(exp_repo.saved) == 1
    assert exp_repo.saved[0].runs[0].parameter_set["transition"] == "NONE->TRAINING"

    registry.promote("ver-cb", ModelStatus.CANDIDATE, notes="Valid threshold validation")
    assert len(exp_repo.saved) == 2
    assert exp_repo.saved[1].runs[0].parameter_set["transition"] == "TRAINING->CANDIDATE"
    assert exp_repo.saved[1].runs[0].parameter_set["notes"] == "Valid threshold validation"
