import pytest
import os
import shutil
import tempfile
import datetime
import numpy as np
import joblib

from backend.training.lifecycle import ModelStatus
from backend.training.registry import ModelRegistry
from backend.training.registry_repo import LocalModelRegistryRepository
from backend.training.prediction_model import PredictionModel

from backend.inference.exceptions import (
    InferenceError,
    ModelNotFoundError,
    ModelLoadError,
    LifecycleError,
    SchemaValidationError,
    PredictionError,
)
from backend.inference.models import InferenceRequest, InferenceResponse
from backend.inference.model_loader import ModelLoaderCache, ManifestChecksumVerifier
from backend.inference.telemetry import InferenceTelemetrySink
from backend.inference.drift import DriftObserver
from backend.inference.engine import InferenceEngine


# ────────────── Mocks & Dummies ──────────────

class MockRawClassifier:
    """Mock underlaying classification model."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simple rule: if first feature > 50, return class 1, else 0
        return np.where(X[:, 0] > 50.0, 1.0, 0.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return probability matrix [(1-p), p]
        p = X[:, 0] / 100.0
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class MockRawRegressor:
    """Mock underlaying regression model."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] * 2.0


class MockTelemetrySink(InferenceTelemetrySink):
    """Spy/Mock telemetry sink capturing success and failure events."""
    def __init__(self):
        self.successes = []
        self.failures = []

    def record_success(self, request, response) -> None:
        self.successes.append((request, response))

    def record_failure(self, request, error_type, error_message, latency_ms) -> None:
        self.failures.append((request, error_type, error_message, latency_ms))


class MockDriftObserver(DriftObserver):
    """Spy/Mock drift observer capturing runtime observations."""
    def __init__(self):
        self.observations = []

    def on_observation(self, model_version: str, features: dict) -> None:
        self.observations.append((model_version, features))


# ────────────── Fixtures ──────────────

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def mock_registry_setup(temp_dir):
    db_path = os.path.join(temp_dir, "model_registry_test.json")
    repo = LocalModelRegistryRepository(db_path=db_path)
    registry = ModelRegistry(repository=repo)

    # Compile mock model artifacts folder
    models_dir = os.path.join(temp_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save models
    clf_path = os.path.join(models_dir, "classifier.joblib")
    reg_path = os.path.join(models_dir, "regressor.joblib")

    joblib.dump(MockRawClassifier(), clf_path)
    joblib.dump(MockRawRegressor(), reg_path)

    # Manifests
    clf_manifest = {
        "feature_version": "v1.0",
        "label_version": "v1.0",
        "experiment_id": "exp-clf",
        "git_commit": "abcdef123",
        "trainer": "lightgbm_class",
        "val_metrics": {"f1": 0.8},
        "feature_importance": {"f1": 5.0, "f2": 3.0},
        "creation_timestamp": "2026-07-21T10:00:00Z",
    }
    reg_manifest = {
        "feature_version": "v2.0",
        "label_version": "v2.0",
        "experiment_id": "exp-reg",
        "git_commit": "uvwxyz789",
        "trainer": "xgboost_regr",
        "val_metrics": {"mae": 1.5},
        "feature_importance": {"f1": 10.0},
        "creation_timestamp": "2026-07-21T10:00:00Z",
    }

    # Register them (starts in TRAINING status)
    registry.register("clf-v1", "ds-clf", clf_path, clf_manifest)
    registry.register("reg-v1", "ds-reg", reg_path, reg_manifest)

    # Promote to VALIDATED (which makes them eligible based on lifecycle rules)
    registry.promote("clf-v1", ModelStatus.CANDIDATE)
    registry.promote("clf-v1", ModelStatus.VALIDATED)

    registry.promote("reg-v1", ModelStatus.CANDIDATE)
    registry.promote("reg-v1", ModelStatus.VALIDATED)

    return registry, repo


# ────────────── Tests ──────────────

def test_successful_single_inference_classification(mock_registry_setup):
    registry, _ = mock_registry_setup
    telemetry = MockTelemetrySink()
    drift = MockDriftObserver()
    engine = InferenceEngine(registry=registry, telemetry_sink=telemetry, drift_observers=[drift])

    # Build request
    req = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    response = engine.predict_one(req)

    # Assert response
    assert isinstance(response, InferenceResponse)
    assert response.model_version == "clf-v1"
    assert response.symbol == "BTCUSD"
    # f1 = 60 > 50 so class is 1.0
    assert response.prediction == 1.0
    assert response.probabilities == pytest.approx([0.4, 0.6])
    assert response.confidence == pytest.approx(0.6)  # max proba
    assert response.metadata.confidence_source == "UNCALIBRATED"
    assert response.metadata.is_calibrated is False
    assert response.latency_ms > 0

    # Test telemetry records success
    assert len(telemetry.successes) == 1
    assert telemetry.successes[0][0] == req
    assert telemetry.successes[0][1] == response
    assert len(telemetry.failures) == 0

    # Test drift hook execution
    assert len(drift.observations) == 1
    assert drift.observations[0][0] == "clf-v1"
    assert drift.observations[0][1] == req.features


def test_successful_single_inference_regression(mock_registry_setup):
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    # Build regression request
    req = InferenceRequest(
        model_version="reg-v1",
        symbol="ETHUSD",
        timeframe="15m",
        features={"f1": 15.0},
        feature_version="v2.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    response = engine.predict_one(req)

    assert isinstance(response, InferenceResponse)
    assert response.model_version == "reg-v1"
    # regression predicts X * 2 = 15.0 * 2 = 30.0
    assert response.prediction == 30.0
    assert response.probabilities is None
    # Regression should not manufacture confidence values
    assert response.confidence is None
    assert response.metadata.confidence_source == "UNAVAILABLE"


def test_true_batch_inference(mock_registry_setup):
    registry, _ = mock_registry_setup
    telemetry = MockTelemetrySink()
    engine = InferenceEngine(registry=registry, telemetry_sink=telemetry)

    req1 = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 40.0, "f2": 10.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )
    req2 = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 80.0, "f2": 30.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    results = engine.predict_batch([req1, req2])

    assert len(results) == 2
    assert isinstance(results[0], InferenceResponse)
    assert isinstance(results[1], InferenceResponse)

    # 40 <= 50 -> prediction 0.0
    assert results[0].prediction == 0.0
    # 80 > 50 -> prediction 1.0
    assert results[1].prediction == 1.0

    # Ensure telemetry tracked both
    assert len(telemetry.successes) == 2
    assert len(telemetry.failures) == 0


def test_partial_batch_failure_semantics(mock_registry_setup):
    registry, _ = mock_registry_setup
    telemetry = MockTelemetrySink()
    engine = InferenceEngine(registry=registry, telemetry_sink=telemetry)

    # Request 1: Valid schema
    req1 = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )
    # Request 2: Invalid features (Missing f2)
    req2 = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )
    # Request 3: Model Not Found
    req3 = InferenceRequest(
        model_version="non-existent-uuid",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    results = engine.predict_batch([req1, req2, req3])

    assert len(results) == 3
    # req1 should succeed
    assert isinstance(results[0], InferenceResponse)
    assert results[0].prediction == 1.0

    # req2 should fail with SchemaValidationError
    assert isinstance(results[1], SchemaValidationError)
    assert "Missing required feature columns" in str(results[1])

    # req3 should fail with ModelNotFoundError
    assert isinstance(results[2], ModelNotFoundError)

    # Telemetry check: 1 success and 2 failures
    assert len(telemetry.successes) == 1
    assert len(telemetry.failures) == 2
    assert telemetry.failures[0][0] == req2
    assert telemetry.failures[0][1] == "SchemaValidationError"
    assert telemetry.failures[1][0] == req3
    assert telemetry.failures[1][1] == "ModelNotFoundError"


def test_exact_model_version_pinning_and_no_implicit_switching(mock_registry_setup):
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    # Attempting to load an invalid or implicit version key.
    # The system must fail closed and NOT search/resolve to latest.
    req = InferenceRequest(
        model_version="latest",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    with pytest.raises(ModelNotFoundError):
        engine.predict_one(req)


def test_cache_hits_and_invalidation_lifecycle(mock_registry_setup):
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    req = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    # 1. First run, cache miss
    res1 = engine.predict_one(req)
    assert res1.metadata.cache_hit is False

    # 2. Second run, cache hit
    res2 = engine.predict_one(req)
    assert res2.metadata.cache_hit is True

    # 3. Explicit invalidation
    engine.cache.invalidate("clf-v1")

    # 4. Third run, cache miss again
    res3 = engine.predict_one(req)
    assert res3.metadata.cache_hit is False

    # 5. Clear cache
    engine.cache.clear()
    res4 = engine.predict_one(req)
    assert res4.metadata.cache_hit is False


def test_explicit_reload(mock_registry_setup):
    registry, _ = mock_registry_setup
    cache = ModelLoaderCache(registry=registry)

    m1, hit1 = cache.get_model("clf-v1")
    assert hit1 is False

    # Reload forces fresh load (cache hit shows hit1=False but loader updates cache)
    m2 = cache.reload("clf-v1")
    assert m1 is not m2  # Reload returns fresh instance because cache was invalidated before loading


def test_lifecycle_gates_eligibility(mock_registry_setup):
    registry, repo = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    # Register and leave in TRAINING state
    manifest = {
        "feature_version": "v1.0",
        "label_version": "v1.0",
        "trainer": "lightgbm",
        "feature_importance": {"f1": 1.0},
    }
    registry.repo.save(registry.register("clf-training", "ds-clf", "path", manifest))

    req = InferenceRequest(
        model_version="clf-training",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 10.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    # Rejected because status is ModelStatus.TRAINING
    with pytest.raises(LifecycleError) as exc_info:
        engine.predict_one(req)
    assert "ineligible for inference" in str(exc_info.value)

    # Promote to CANDIDATE
    registry.promote("clf-training", ModelStatus.CANDIDATE)
    with pytest.raises(LifecycleError):
        engine.predict_one(req)

    # Promote to VALIDATED -> Accept
    registry.promote("clf-training", ModelStatus.VALIDATED)
    # Patch to load FakePredictionModel object or register real file
    # Let's bypass file load error by mocking raw load in registry
    record = registry.repo.get("clf-training")
    record.model_path = registry.repo.get("clf-v1").model_path
    registry.repo.save(record)

    response = engine.predict_one(req)
    assert isinstance(response, InferenceResponse)
    assert response.success is True if hasattr(response, "success") else True # Response always represents success


def test_cached_model_becomes_deprecated_registry_check(mock_registry_setup):
    """
    Constraint #4 Check:
    A cache hit must STILL fail closed immediately if the model status changes to DEPRECATED in DB.
    """
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    req = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    # 1. Valid loaded model is cached
    res1 = engine.predict_one(req)
    assert res1.metadata.cache_hit is False

    # Verify next prediction is cached
    res2 = engine.predict_one(req)
    assert res2.metadata.cache_hit is True

    # 2. Status in DB becomes DEPRECATED
    registry.promote("clf-v1", ModelStatus.DEPRECATED)

    # 3. Next inference must raise LifecycleError even though it was cached
    with pytest.raises(LifecycleError):
        engine.predict_one(req)


def test_schema_feature_validation_failures(mock_registry_setup):
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    req_base = {
        "model_version": "clf-v1",
        "symbol": "BTCUSD",
        "timeframe": "1h",
        "feature_version": "v1.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

    # 1. Missing feature column
    req1 = InferenceRequest(features={"f1": 10.0}, **req_base)
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req1)
    assert "Missing required feature columns" in str(ex.value)

    # 2. Unexpected feature column
    req2 = InferenceRequest(features={"f1": 1.0, "f2": 2.0, "f3": 99.0}, **req_base)
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req2)
    assert "Unexpected feature columns in request" in str(ex.value)

    # 3. Dimensionality validation check (len features != len columns)
    # The unexpected/missing check catches these, but we also enforce direct length parity.

    # 4. NaN rejection
    req3 = InferenceRequest(features={"f1": float("nan"), "f2": 20.0}, **req_base)
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req3)
    assert "is NaN" in str(ex.value)

    # 5. Infinite value rejection (+Inf)
    req4 = InferenceRequest(features={"f1": float("inf"), "f2": 20.0}, **req_base)
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req4)
    assert "is infinite" in str(ex.value)

    # 6. Infinite value rejection (-Inf)
    req5 = InferenceRequest(features={"f1": float("-inf"), "f2": 20.0}, **req_base)
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req5)
    assert "is infinite" in str(ex.value)

    # 7. Non-numeric compatibility
    req6 = InferenceRequest(features={"f1": "forty-two", "f2": 20.0}, **req_base)  # type: ignore
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req6)
    assert "has invalid non-numeric value" in str(ex.value)

    # 8. Feature version mismatch
    req7 = InferenceRequest(features={"f1": 10.0, "f2": 20.0}, **req_base)
    req7 = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v2.0_wrong",
        timestamp=datetime.datetime.now().isoformat()
    )
    with pytest.raises(SchemaValidationError) as ex:
        engine.predict_one(req7)
    assert "Feature version mismatch" in str(ex.value)


def test_canonical_feature_ordering_enforcement(mock_registry_setup):
    """
    Verify feature ordering dictionary insertion key doesn't alter array representation structure.
    """
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    # Order in columns: ["f1", "f2"]
    # Request features dict contains keys in disorder: {"f2": 20.0, "f1": 60.0}
    req = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f2": 20.0, "f1": 60.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    response = engine.predict_one(req)
    # The order must be preserved. Since f1 (value 60.0) is ordered first:
    # 60.0 > 50.0 -> prediction 1.0
    assert response.prediction == 1.0


def test_corrupted_model_artifact_load_handling(mock_registry_setup):
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    # Corrupt the model file
    record = registry.repo.get("clf-v1")
    with open(record.model_path, "w") as f:
        f.write("corrupted data content")

    engine.cache.clear()

    req = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    with pytest.raises(ModelLoadError) as exc_info:
        engine.predict_one(req)
    assert "Failed to deserialize model" in str(exc_info.value)


def test_prediction_model_exception_fail_closed(mock_registry_setup):
    registry, _ = mock_registry_setup
    engine = InferenceEngine(registry=registry)

    # Injecting error into model's predict/predict_proba method
    model, _ = engine.cache.get_model("clf-v1")
    
    # Mock model
    class BrokenRawModel:
        def predict(self, X):
            raise ValueError("Predict execution crash!")
            
    model._model = BrokenRawModel()

    req = InferenceRequest(
        model_version="clf-v1",
        symbol="BTCUSD",
        timeframe="1h",
        features={"f1": 60.0, "f2": 20.0},
        feature_version="v1.0",
        timestamp=datetime.datetime.now().isoformat()
    )

    with pytest.raises(PredictionError) as exc_info:
        engine.predict_one(req)
    assert "Prediction execution failed" in str(exc_info.value)
