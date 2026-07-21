import pytest
import os
import shutil
import tempfile
import datetime
import pandas as pd
import numpy as np
from typing import Any

from backend.models.candle import Candle
from backend.training.lifecycle import ModelStatus
from backend.training.registry import ModelRegistry
from backend.training.registry_repo import LocalModelRegistryRepository
from backend.training.prediction_model import PredictionModel
from backend.training.validation import (
    ValidationPipeline,
    ValidationConfig,
    ValidationResult,
    ValidationPaperExecutor,
)
from backend.training.validation_report import generate_validation_report
from backend.strategy.registry import StrategyLoader


# ────────────── Helpers & Mocks ──────────────

class FakePredictionModel(PredictionModel):
    def __init__(self):
        super().__init__(model=self, model_type="fake", task_type="classification", feature_columns=["rsi", "ema20", "close"])

    def predict(self, features: Any) -> Any:
        if isinstance(features, dict):
            return float(1 if features.get("rsi", 50.0) <= 30.0 else 0)
        if isinstance(features, np.ndarray):
            if features.ndim == 2:
                return np.where(features[:, 0] <= 30.0, 1.0, 0.0)
            return float(1 if features[0] <= 30.0 else 0)
        return 0.0


@pytest.fixture
def clean_registry():
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "model_registry_test.json")
    repo = LocalModelRegistryRepository(db_path=db_path)
    registry = ModelRegistry(repository=repo)
    yield registry, repo
    shutil.rmtree(d)


def _generate_mock_candles(n: int = 50) -> list:
    candles = []
    base_time = datetime.datetime(2026, 7, 1, 10, 0)
    for i in range(n):
        # price series going up/down
        price = 100.0 + (i % 5) * 5
        c = Candle(
            symbol="BTCUSD",
            open_time=base_time + datetime.timedelta(minutes=i),
            open=price,
            high=price + 2,
            low=price - 2,
            close=price,
            volume=100.0
        )
        candles.append(c)
    return candles


def _generate_mock_df(n: int = 100) -> pd.DataFrame:
    # Generates a feature dataset matching Mock columns
    np.random.seed(42)
    return pd.DataFrame({
        "rsi": np.random.uniform(10, 90, n),
        "ema20": np.random.uniform(90, 110, n),
        "close": np.random.uniform(90, 110, n),
        "label": np.random.choice([0.0, 1.0], n),
    })


# ────────────── Tests ──────────────

def test_benchmark_registry_loading():
    """Verify that all baseline strategies load successfully via StrategyLoader."""
    benchmarks = ["buy_and_hold", "always_flat", "random_predictor", "ema_crossover", "rsi_strategy", "macd_strategy"]
    for bench in benchmarks:
        strategy = StrategyLoader.load_strategy(bench)
        assert strategy is not None


def test_custom_validator_paper_executor_applies_commissions_and_slippage():
    """Verify that ValidationPaperExecutor applies slippage and commission modifiers correctly."""
    class FakeSignal:
        def __init__(self):
            self.action = "BUY"
            self.confidence = 0.9
            self.reason = "test_signal"

    class FakeRepository:
        def __init__(self):
            self.created = []
        def create(self, t):
            self.created.append(t)
        def get_open_trade(self, s):
            return self.created[-1] if self.created else None
        def update(self, t):
            pass

    repo = FakeRepository()
    executor = ValidationPaperExecutor(
        db_session=None,
        clock=None,
        commission_pct=0.01,  # 1% commission
        slippage_pct=0.02,    # 2% slippage
    )
    # Mock self.trade_repository to bypass DB operations
    executor.trade_repository = repo  # type: ignore

    candle = Candle(
        symbol="BTCUSD",
        open_time=datetime.datetime.now(),
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        volume=10.0
    )

    # 1. Test Entry
    executor.execute(FakeSignal(), candle)
    assert len(executor.portfolio.get_open_positions()) == 1
    pos = executor.portfolio.get_open_positions()[0]
    
    # entry_price = 100 * (1 + 0.02) = 102.0
    assert pos.entry_price == pytest.approx(102.0)
    assert len(repo.created) == 1
    assert repo.created[0].entry_price == pytest.approx(102.0)

    # 2. Test Exit on Stop Loss / Take Profit
    # PnL = (exit_price - entry_price) * qty
    # Set candle exit close price high enough to exceed TAKE_PROFIT threshold
    candle_exit = Candle(
        symbol="BTCUSD",
        open_time=datetime.datetime.now(),
        open=200.0,
        high=200.0,
        low=200.0,
        close=200.0,
        volume=10.0
    )
    executor.TAKE_PROFIT = 5.0
    
    executor.execute(None, candle_exit)
    assert len(executor.portfolio.get_open_positions()) == 0
    trade = repo.created[0]
    assert trade.status == "CLOSED"
    # exit_price = 200 * (1 - 0.02) = 196.0
    assert trade.exit_price == pytest.approx(196.0)
    
    # total volume = (102.0 + 196.0) * qty = 298.0 * qty
    # commission = 298.0 * qty * 0.01 = 2.98 * qty
    assert trade.commission > 0.0
    assert trade.pnl == pytest.approx(((196.0 - 102.0) * pos.quantity) - trade.commission)


def test_validation_pipeline_fully_concludes(clean_registry):
    """Verify that ValidationPipeline executes all stages and updates status."""
    registry, repo = clean_registry

    # Persist mock model object inside version path
    imported_models_dir = tempfile.mkdtemp()
    model_version = "ver-validation-test"
    model_dir = os.path.join(imported_models_dir, model_version)
    os.makedirs(model_dir, exist_ok=True)
    
    import joblib
    fake_model = FakePredictionModel()
    joblib.dump(fake_model, os.path.join(model_dir, "model.joblib"))

    # Register model in Candidate status
    manifest = {
        "feature_version": "v1",
        "label_version": "v1",
        "experiment_id": "exp-val",
        "git_commit": "abcdef",
        "trainer": "lightgbm",
        "val_metrics": {"f1": 0.65},
        "feature_importance": {"rsi": 5.0, "ema20": 2.0, "close": 1.0},
        "creation_timestamp": "2026-07-20T12:00:00Z",
    }
    registry.register(
        model_version=model_version,
        dataset_version="ds-val",
        model_path=os.path.join(model_dir, "model.joblib"),
        manifest=manifest,
    )
    registry.promote(model_version, ModelStatus.CANDIDATE)

    pipeline = ValidationPipeline(model_registry=registry)
    
    candles = _generate_mock_candles(30)
    df_features = _generate_mock_df()

    # Run validation under high stability threshold to see if it rejects or accepts
    config = ValidationConfig(
        min_sharpe=-10.0,  # Ensure WF passes
        min_f1=0.50,
        max_drawdown=0.90,
        min_return=-10000.0,
        stability_threshold=0.0  # Force stability approval
    )

    result = pipeline.validate(
        model_version=model_version,
        dataset_df=df_features,
        candles=candles,
        config=config
    )

    # Check validation results
    assert isinstance(result, ValidationResult)
    assert result.validation_decision in ["APPROVED", "REJECTED"]
    assert result.calibration_status == "READY_INTERFACE_PLATT_ISOTONIC"
    assert "rsi" in result.drift_baseline
    assert "mean" in result.drift_baseline["rsi"]
    assert "quantiles" in result.drift_baseline["rsi"]

    # Verify report is built
    report = generate_validation_report(
        model_version=model_version,
        dataset_version="ds-val",
        experiment_id="exp-val",
        result=result,
        reviewer_notes="Automatic validation"
    )
    assert "Institutional Model Validation Report" in report
    assert model_version in report
    assert "Purged Time-Series Cross Validation Summary" in report

    # Clean up temp model folder
    shutil.rmtree(imported_models_dir)


from backend.training.validation import BaseTimeSeriesSplitter, PurgedTimeSeriesSplit

def test_purged_time_series_split_chronological_and_bounds():
    df = _generate_mock_df(100)
    splitter = PurgedTimeSeriesSplit(n_splits=5, purge_horizon=5, embargo_size=5)
    splits = splitter.split(df)
    
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        
        # Chronological ordering verification
        assert np.all(np.diff(train_idx) >= 1) or len(train_idx) <= 1
        assert np.all(np.diff(val_idx) >= 1) or len(val_idx) <= 1
        
        # Validation indices must be contiguous
        assert val_idx[-1] - val_idx[0] == len(val_idx) - 1


def test_purged_time_series_split_purging_and_embargo():
    df = _generate_mock_df(100)
    H = 8
    E = 6
    splitter = PurgedTimeSeriesSplit(n_splits=4, purge_horizon=H, embargo_size=E)
    splits = splitter.split(df)
    
    for train_idx, val_idx in splits:
        v_start = val_idx[0]
        v_end = val_idx[-1] + 1
        
        # Check purging: no training sample inside [v_start - H, v_start - 1]
        purged_range = range(max(0, v_start - H), v_start)
        for idx in purged_range:
            assert idx not in train_idx
            
        # Check embargo: no training sample inside [v_end, v_end + E - 1]
        embargo_range = range(v_end, min(100, v_end + E))
        for idx in embargo_range:
            assert idx not in train_idx


def test_purged_time_series_split_small_dataset():
    # Graceful handling of small dataset
    df = _generate_mock_df(3)
    splitter = PurgedTimeSeriesSplit(n_splits=5, purge_horizon=2, embargo_size=2)
    splits = splitter.split(df)
    assert len(splits) == 0


def test_purged_time_series_split_zero_horizon():
    df = _generate_mock_df(50)
    splitter = PurgedTimeSeriesSplit(n_splits=5, purge_horizon=0, embargo_size=0)
    splits = splitter.split(df)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) + len(val_idx) == 50


def test_purged_time_series_split_determinism():
    df = _generate_mock_df(100)
    s1 = PurgedTimeSeriesSplit(n_splits=5, purge_horizon=5, embargo_size=5)
    s2 = PurgedTimeSeriesSplit(n_splits=5, purge_horizon=5, embargo_size=5)
    
    splits1 = s1.split(df)
    splits2 = s2.split(df)
    
    assert len(splits1) == len(splits2)
    for (t1, v1), (t2, v2) in zip(splits1, splits2):
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)


def test_purged_time_series_split_leakage_prevention_comparison():
    # Simulate a target leakage setup: y_t depends on future value x_{t+2}
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(n - 2):
        y[t] = 1.0 if x[t + 2] > 0.0 else 0.0
        
    df = pd.DataFrame({"feat": x, "label": y})
    
    # Validation boundary at index 50
    # A standard index split at 50 would set train=[0..49], val=[50..100]
    # In this case: train sample at t=48 leaks x_50 (first validation point)!
    # And train sample at t=49 leaks x_51 (second validation point)!
    
    # With PurgedTimeSeriesSplit (purge_horizon=2):
    # Train samples within range [50-2, 49] = [48, 49] are purged.
    splitter = PurgedTimeSeriesSplit(n_splits=2, purge_horizon=2, embargo_size=0)
    splits = splitter.split(df)
    
    # First split validation starts at index 50
    train_idx, val_idx = splits[1]
    assert val_idx[0] == 50
    # Training elements before validation end at 47 (48 and 49 are purged)
    assert 48 not in train_idx
    assert 49 not in train_idx
    assert 47 in train_idx


def test_validation_pipeline_cv_rejections_and_data(clean_registry):
    # Verify that ValidationPipeline runs CV and records splitter info
    registry, repo = clean_registry
    
    imported_models_dir = tempfile.mkdtemp()
    model_version = "ver-cv-test"
    model_dir = os.path.join(imported_models_dir, model_version)
    os.makedirs(model_dir, exist_ok=True)
    
    import joblib
    fake_model = FakePredictionModel()
    joblib.dump(fake_model, os.path.join(model_dir, "model.joblib"))
    
    # Save a fake metadata file to test metadata detection
    dataset_version = "ds-cv-test"
    dataset_dir = os.path.join("data/datasets", dataset_version)
    os.makedirs(dataset_dir, exist_ok=True)
    meta_data = {
        "label_horizon": 5,
        "generation_parameters": {"horizon": 5}
    }
    import json
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(meta_data, f)
        
    manifest = {
        "feature_version": "v1",
        "label_version": "v1",
        "experiment_id": "exp-cv",
        "git_commit": "abcdef",
        "trainer": "lightgbm",
        "val_metrics": {"f1": 0.35},
        "feature_importance": {"rsi": 5.0, "ema20": 2.0, "close": 1.0},
        "creation_timestamp": "2026-07-20T12:00:00Z",
    }
    registry.register(
        model_version=model_version,
        dataset_version=dataset_version,
        model_path=os.path.join(model_dir, "model.joblib"),
        manifest=manifest,
    )
    registry.promote(model_version, ModelStatus.CANDIDATE)
    
    pipeline = ValidationPipeline(model_registry=registry)
    candles = _generate_mock_candles(30)
    df_features = _generate_mock_df()
    
    # Run with min_f1=0.90 to force CV failure
    config = ValidationConfig(
        min_sharpe=-10.0,
        min_f1=0.90,
        max_drawdown=0.90,
        min_return=-10000.0,
        stability_threshold=0.0,
        n_splits=3,
        embargo_size=5
    )
    
    result = pipeline.validate(
        model_version=model_version,
        dataset_df=df_features,
        candles=candles,
        config=config
    )
    
    assert isinstance(result, ValidationResult)
    # Decision must be REJECTED because average CV score will not beat 0.90
    assert result.validation_decision == "REJECTED"
    assert result.splitter_info["splitter_type"] == "PurgedTimeSeriesSplit"
    assert result.splitter_info["n_splits"] == 3
    assert result.splitter_info["purge_size"] == 5  # Derived from DatasetMetadata!
    assert result.splitter_info["embargo_size"] == 5
    assert result.splitter_info["total_samples_purged"] > 0
    assert result.splitter_info["total_samples_embargoed"] > 0
    assert "average_cv_score" in result.cv_metrics
    
    # Clean up
    shutil.rmtree(imported_models_dir)
    shutil.rmtree(dataset_dir)
