import pytest
import os
import shutil
import tempfile
from datetime import datetime, timedelta

from backend.dataset.labelers import (
    FutureReturnClassification,
    FutureReturnRegression,
    TPSLOutcomeLabeler,
)
from backend.dataset.models import DatasetMetadata
from backend.dataset.builder import DatasetBuilder


class FakeCandle:
    """Lightweight candle stub for unit tests."""

    def __init__(self, close, high=None, low=None, symbol="BTCUSD", timeframe="5m", open_time=None):
        self.close = close
        self.high = high if high is not None else close * 1.01
        self.low = low if low is not None else close * 0.99
        self.open = close
        self.volume = 100.0
        self.symbol = symbol
        self.timeframe = timeframe
        self.open_time = open_time or datetime(2026, 1, 1)


def _make_candle_series(prices, symbol="BTCUSD"):
    """Generate a list of FakeCandles from a price list."""
    base_time = datetime(2026, 1, 1)
    candles = []
    for i, p in enumerate(prices):
        candles.append(
            FakeCandle(
                close=p,
                high=p * 1.02,
                low=p * 0.98,
                symbol=symbol,
                open_time=base_time + timedelta(minutes=i * 5),
            )
        )
    return candles


# ────────────── Labeling Tests ──────────────


def test_future_return_classification_up():
    prices = [100.0] * 5 + [120.0]
    candles = _make_candle_series(prices)
    labeler = FutureReturnClassification()
    label = labeler.label(candles, 0, {"horizon": 5, "threshold": 0.01})
    assert label == 1.0


def test_future_return_classification_down():
    prices = [100.0] * 5 + [80.0]
    candles = _make_candle_series(prices)
    labeler = FutureReturnClassification()
    label = labeler.label(candles, 0, {"horizon": 5, "threshold": 0.01})
    assert label == -1.0


def test_future_return_classification_neutral():
    prices = [100.0] * 6
    candles = _make_candle_series(prices)
    labeler = FutureReturnClassification()
    label = labeler.label(candles, 0, {"horizon": 5, "threshold": 0.01})
    assert label == 0.0


def test_future_return_classification_none_at_end():
    prices = [100.0] * 3
    candles = _make_candle_series(prices)
    labeler = FutureReturnClassification()
    label = labeler.label(candles, 2, {"horizon": 5, "threshold": 0.01})
    assert label is None


def test_future_return_regression():
    prices = [100.0] * 5 + [110.0]
    candles = _make_candle_series(prices)
    labeler = FutureReturnRegression()
    label = labeler.label(candles, 0, {"horizon": 5})
    assert label == pytest.approx(0.10, abs=0.001)


def test_tpsl_tp_hit():
    # TP at +2%, SL at -1%. Price goes up immediately.
    candles = _make_candle_series([100.0, 103.0, 105.0])
    # Adjust high so TP threshold is hit
    candles[1].high = 103.0
    labeler = TPSLOutcomeLabeler()
    label = labeler.label(candles, 0, {"tp_pct": 0.02, "sl_pct": 0.01, "horizon": 5})
    assert label == 1.0


def test_tpsl_sl_hit():
    # Price drops below SL
    candles = _make_candle_series([100.0, 98.0, 97.0])
    candles[1].low = 98.0
    labeler = TPSLOutcomeLabeler()
    label = labeler.label(candles, 0, {"tp_pct": 0.05, "sl_pct": 0.01, "horizon": 5})
    assert label == 0.0


# ────────────── Metadata Tests ──────────────


def test_metadata_serialization():
    meta = DatasetMetadata(
        dataset_id="ds-1",
        generation_timestamp="2026-07-20T12:00:00Z",
        source_symbols=["BTCUSD"],
        timeframe="5m",
        sample_count=100,
        feature_list=["rsi", "ema20"],
        labeling_strategy="future_return_classification",
    )
    d = meta.to_dict()
    restored = DatasetMetadata.from_dict(d)
    assert restored.dataset_id == "ds-1"
    assert restored.sample_count == 100
    assert restored.feature_list == ["rsi", "ema20"]


def test_metadata_version_uniqueness():
    m1 = DatasetMetadata()
    m2 = DatasetMetadata()
    assert m1.dataset_id != m2.dataset_id


# ────────────── DatasetBuilder Tests ──────────────


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_dataset_builder_generation(temp_dir):
    """Generate a dataset from synthetic candles and verify Parquet output."""
    # Need MIN_CANDLES candles (default 50) + horizon for labels
    prices = [100.0 + i * 0.5 for i in range(80)]
    candles = _make_candle_series(prices)

    labeler = FutureReturnClassification()
    builder = DatasetBuilder(
        labeler=labeler,
        label_params={"horizon": 3, "threshold": 0.005},
        output_dir=temp_dir,
    )

    metadata, df = builder.build(candles, dataset_name="test_ds")

    assert len(df) > 0
    assert "label" in df.columns
    assert "split" in df.columns
    assert metadata.sample_count == len(df)
    assert metadata.labeling_strategy == "future_return_classification"

    # Check Parquet file exists
    version_dir = os.path.join(temp_dir, f"test_ds_{metadata.dataset_id}")
    assert os.path.isfile(os.path.join(version_dir, "dataset.parquet"))
    assert os.path.isfile(os.path.join(version_dir, "metadata.json"))


def test_dataset_builder_splits(temp_dir):
    """Verify train/val/test counts approximately match ratios."""
    prices = [100.0 + i * 0.3 for i in range(80)]
    candles = _make_candle_series(prices)

    labeler = FutureReturnRegression()
    builder = DatasetBuilder(
        labeler=labeler,
        label_params={"horizon": 2},
        output_dir=temp_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    metadata, df = builder.build(candles, dataset_name="split_ds")

    total = len(df)
    train_count = len(df[df["split"] == "train"])
    val_count = len(df[df["split"] == "val"])
    test_count = len(df[df["split"] == "test"])

    assert train_count + val_count + test_count == total
    assert metadata.train_count == train_count
    assert metadata.val_count == val_count
    assert metadata.test_count == test_count


def test_dataset_builder_csv_export(temp_dir):
    """Verify CSV export when flag is set."""
    prices = [100.0 + i for i in range(80)]
    candles = _make_candle_series(prices)

    labeler = FutureReturnRegression()
    builder = DatasetBuilder(
        labeler=labeler,
        label_params={"horizon": 2},
        output_dir=temp_dir,
    )

    metadata, df = builder.build(candles, dataset_name="csv_ds", export_csv=True)

    version_dir = os.path.join(temp_dir, f"csv_ds_{metadata.dataset_id}")
    assert os.path.isfile(os.path.join(version_dir, "dataset.csv"))
