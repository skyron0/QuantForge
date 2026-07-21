import os
import tempfile
import pytest
import numpy as np
from typing import Dict, Any, List

from backend.training.lifecycle import ModelStatus
from backend.training.registry import ModelRegistry
from backend.training.registry_models import RegisteredModel
from backend.inference.exceptions import ArtifactIntegrityError, PredictionError
from backend.inference.integrity import (
    ArtifactIntegrityVerifier,
    IntegrityPolicy,
    calculate_sha256,
    IntegrityState,
)
from backend.inference.calibration import (
    PlattCalibrator,
    IsotonicCalibrator,
    calculate_ece,
    evaluate_calibration_metrics,
)
from backend.inference.drift_detector import (
    FeatureDriftDetector,
    DriftStatus,
    DriftMetric,
    DriftThresholds,
)
from backend.inference.drift import ActiveDriftObserver


# ────────────── 1. Test SHA-256 Integrity Verification ──────────────

def test_calculate_sha256():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"QuantForge-test-content")
        f_path = f.name
    
    try:
        checksum = calculate_sha256(f_path)
        assert len(checksum) == 64
        # Re-calc matches
        assert calculate_sha256(f_path) == checksum
    finally:
        os.remove(f_path)


def test_integrity_verifier_policies():
    verifier = ArtifactIntegrityVerifier()
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"QuantForge weight file")
        f_path = f.name
    
    try:
        actual_sha = calculate_sha256(f_path)
        
        # 1. Success matching
        assert verifier.verify(f_path, actual_sha, ModelStatus.PRODUCTION) == IntegrityState.VERIFIED
        
        # 2. Mismatch ALWAYS FAILS
        wrong_sha = "a" * 64
        with pytest.raises(ArtifactIntegrityError):
            verifier.verify(f_path, wrong_sha, ModelStatus.PRODUCTION)
            
        # 3. Missing checksum under STRICT policy
        with pytest.raises(ArtifactIntegrityError):
            verifier.verify(f_path, None, ModelStatus.VALIDATED, policy=IntegrityPolicy.STRICT)
            
        # 4. Missing checksum under PRODUCTION_STRICT policy (fails for PRODUCTION, accepts for CANDIDATE/VALIDATED)
        with pytest.raises(ArtifactIntegrityError):
            verifier.verify(f_path, None, ModelStatus.PRODUCTION, policy=IntegrityPolicy.PRODUCTION_STRICT)
            
        assert verifier.verify(f_path, None, ModelStatus.VALIDATED, policy=IntegrityPolicy.PRODUCTION_STRICT) == IntegrityState.UNVERIFIED_LEGACY
        
        # 5. Missing checksum under LENIENT policy
        assert verifier.verify(f_path, None, ModelStatus.PRODUCTION, policy=IntegrityPolicy.LENIENT) == IntegrityState.UNVERIFIED_LEGACY
        
    finally:
        os.remove(f_path)


# ────────────── 2. Test Probability Calibration & ECE ──────────────

def test_platt_calibration():
    # Setup synthethic raw predictions and binary labels
    np.random.seed(42)
    y_raw = np.random.uniform(0.1, 0.9, 100)
    y_labels = (y_raw > 0.5).astype(int)
    
    calibrator = PlattCalibrator()
    calibrator.fit(y_raw, y_labels)
    
    cal_probs = calibrator.transform(y_raw)
    assert cal_probs.shape == (100, 2)
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)
    
    # Save & reload validation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
        temp_path = f.name
    
    try:
        calibrator.save(temp_path)
        reloaded = PlattCalibrator.load(temp_path)
        reloaded_probs = reloaded.transform(y_raw)
        assert np.allclose(cal_probs, reloaded_probs)
    finally:
        os.remove(temp_path)


def test_isotonic_calibration():
    np.random.seed(42)
    y_raw = np.random.uniform(0.01, 0.99, 100)
    y_labels = (y_raw > 0.55).astype(int)
    
    calibrator = IsotonicCalibrator()
    calibrator.fit(y_raw, y_labels)
    
    cal_probs = calibrator.transform(y_raw)
    assert cal_probs.shape == (100, 2)
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)


def test_evaluation_metrics():
    # Deterministic test for calculations
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    # Positive class probabilities are: 0.2, 0.7, 0.1, 0.6
    labels = np.array([0, 1, 0, 1])
    
    metrics = evaluate_calibration_metrics(probs, labels)
    
    # Brier Score = ((0.2-0)^2 + (0.7-1)^2 + (0.1-0)^2 + (0.6-1)^2) / 4 = (0.04 + 0.09 + 0.01 + 0.16) / 4 = 0.3 / 4 = 0.075
    assert abs(metrics["brier_score"] - 0.075) < 1e-5
    assert metrics["log_loss"] > 0
    assert metrics["ece"] >= 0.0


# ────────────── 3. Test Drift Detection & Active monitoring ──────────────

def test_feature_drift_calculations():
    detector = FeatureDriftDetector(thresholds=DriftThresholds(psi_warning=0.10, psi_critical=0.25))
    
    # Generate reference distribution and calculate its actual quantiles dynamically
    np.random.seed(42)
    ref_data = np.random.normal(0, 0.5, 1000)
    q25 = float(np.percentile(ref_data, 25))
    q50 = float(np.percentile(ref_data, 50))
    q75 = float(np.percentile(ref_data, 75))
    
    baseline_stats = {
        "binning_method": "quantile",
        "bin_edges": ["-inf", q25, q50, q75, "inf"],
        "expected_proportions": [0.25, 0.25, 0.25, 0.25],
        "raw_reference": ref_data.tolist()
    }
    
    # 1. No drift runtime observation (similar distribution)
    obs_stable = np.random.normal(0, 0.5, 800)
    res_psi = detector.calculate_psi("feat1", baseline_stats, obs_stable)
    assert res_psi.status == DriftStatus.STABLE
    assert res_psi.score is not None
    assert res_psi.score < 0.10
    
    # 2. Critical drift runtime (shifted mean distribution)
    obs_drifted = np.random.normal(1.5, 0.5, 800)
    res_drift_psi = detector.calculate_psi("feat1", baseline_stats, obs_drifted)
    assert res_drift_psi.status == DriftStatus.CRITICAL
    assert res_drift_psi.score is not None
    assert res_drift_psi.score >= 0.25
    
    # 3. Kolmogorov-Smirnov Test
    res_ks_stable = detector.calculate_ks("feat1", baseline_stats, obs_stable)
    assert res_ks_stable.status == DriftStatus.STABLE
    
    res_ks_drift = detector.calculate_ks("feat1", baseline_stats, obs_drifted)
    assert res_ks_drift.status == DriftStatus.CRITICAL
    assert res_ks_drift.p_value is not None
    assert res_ks_drift.p_value < 0.05


class MockModelRecord:
    def __init__(self, key: str, drift_baseline: dict):
        self.model_version = key
        self.drift_baseline = drift_baseline


class MockModelRepo:
    def __init__(self):
        self.db = {}
        
    def get(self, key):
        return self.db.get(key)
        
    def save(self, obj):
        self.db[obj.model_version] = obj


class MockRegistry:
    def __init__(self):
        self.repo = MockModelRepo()


def test_active_drift_observer():
    mock_reg = MockRegistry()
    baseline = {
        "feat1": {
            "bin_edges": ["-inf", 0.0, "inf"],
            "expected_proportions": [0.5, 0.5]
        }
    }
    record = MockModelRecord("v1_mock", baseline)
    mock_reg.repo.save(record)
    
    # Bounded samples trigger check at 10 observations
    observer = ActiveDriftObserver(registry=mock_reg, window_size=20, min_samples=10)  # type: ignore
    
    # Attach callback hook
    reports = []
    observer.callbacks.append(lambda r: reports.append(r))
    
    # Push 9 samples -> should not trigger check
    for _ in range(9):
        observer.on_observation("v1_mock", {"feat1": -1.2})
    assert len(reports) == 0
    
    # Push 10th sample -> triggers report creation
    observer.on_observation("v1_mock", {"feat1": -0.8})
    assert len(reports) == 1
    assert reports[0].model_version == "v1_mock"
    assert reports[0].total_features_evaluated == 1
