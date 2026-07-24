import pytest
import os
import tempfile
import csv
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from backend.replay.clock import ReplayClock
from backend.replay.exceptions import ReplayValidationError, ReplayInvariantError, DatasetLoadingError
from backend.replay.dataset import validate_ohlc_record, sort_records_deterministically, generate_dataset_hash
from backend.replay.loader import CSVHistoricalCandleLoader
from backend.replay.scheduler import ReplayScheduler
from backend.replay.service import HistoricalReplayService
from backend.replay.models import ReplaySessionConfig, ReplayStatus
from backend.replay.policy import ReplayPolicy

def test_replay_clock_utc_awareness():
    clock = ReplayClock("2023-01-01T00:00:00Z")
    dt = clock.now()
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2023
    assert dt.month == 1
    assert dt.day == 1

def test_replay_clock_advancement():
    clock = ReplayClock("2023-01-01T00:00:00Z")
    clock.set_time("2023-01-01T00:05:00Z")
    assert clock.now().minute == 5

    # Attempting to move backward should raise ValueError
    with pytest.raises(ValueError):
        clock.advance_to("2023-01-01T00:04:00Z")

def test_dataset_ohlc_validation():
    valid_record = {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 10.0,
        "high": 12.0,
        "low": 9.0,
        "close": 11.0,
        "volume": 100.0
    }
    # Should not raise exception
    validate_ohlc_record(valid_record)

    # High < Low should raise ReplayValidationError
    invalid_record = dict(valid_record, high=8.0)
    with pytest.raises(ReplayValidationError):
        validate_ohlc_record(invalid_record)

    # Empty date/timestamp should raise ReplayValidationError
    no_time_record = dict(valid_record, timestamp="")
    with pytest.raises(ReplayValidationError):
        validate_ohlc_record(no_time_record)

def test_generate_dataset_hash():
    records = [
        {"timestamp": "2023-01-01T00:00:00Z", "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0},
        {"timestamp": "2023-01-01T00:01:00Z", "open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0, "volume": 200.0}
    ]
    hash1 = generate_dataset_hash(records)
    hash2 = generate_dataset_hash(records)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string length

def test_csv_historical_candle_loader_missing():
    loader = CSVHistoricalCandleLoader("non_existent_file.csv", "BTC/USDT")
    with pytest.raises(DatasetLoadingError):
        loader.load()

def test_csv_historical_candle_loader_headers():
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Open Time", "Open", "High", "Low", "Close", "Volume"])
            writer.writerow(["2023-01-01T00:00:00Z", "10", "12", "9", "11", "100"])

        loader = CSVHistoricalCandleLoader(path, "BTC/USDT")
        records = loader.load()
        assert len(records) == 1
        assert float(records[0]["open"]) == 10.0
    finally:
        os.close(fd)
        os.remove(path)

def test_replay_scheduler_stepping():
    clock = ReplayClock("2023-01-01T00:00:00Z")
    dataset = [
        {"timestamp": "2023-01-01T00:01:00Z", "open": 10},
        {"timestamp": "2023-01-01T00:02:00Z", "open": 11}
    ]
    called_records = []
    def on_step(rec):
        called_records.append(rec)

    scheduler = ReplayScheduler(clock, dataset, on_step)
    assert scheduler.total_steps == 2
    assert scheduler.current_index == 0

    assert scheduler.step() is True
    assert clock.now().minute == 1
    assert len(called_records) == 1
    assert scheduler.current_index == 1

    assert scheduler.step() is True
    assert clock.now().minute == 2
    assert len(called_records) == 2
    assert scheduler.current_index == 2

    # Data exhausted
    assert scheduler.step() is False

def test_replay_service_policy_validation():
    # If policy.paper_only is false (which violates basic ReplayPolicy structure limit since replays are paper-only),
    # it should raise ReplayValidationError or ReplayInvariantError
    policy = ReplayPolicy(paper_only=False)
    with pytest.raises(ReplayValidationError):
        policy.validate()

def test_historical_replay_service_execution():
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            # Load at least 2 records for buffer minimum warmup requirement
            writer.writerow(["2023-01-01T00:00:00Z", "10.0", "12.0", "9.0", "11.0", "100.0"])
            writer.writerow(["2023-01-01T00:01:00Z", "11.0", "13.0", "10.0", "12.0", "150.0"])
            writer.writerow(["2023-01-01T00:02:00Z", "12.0", "14.0", "11.0", "13.0", "200.0"])

        config = ReplaySessionConfig(
            initial_capital=100000.0,
            max_cycles=100,
            seed=42,
            dataset_path=path,
            enabled_symbols=["BTC/USDT"],
            timeframe="1m"
        )

        service = HistoricalReplayService()
        result = service.run_replay_session(config)

        assert result.status == ReplayStatus.COMPLETED
        assert result.dataset_metadata.row_count == 3
        assert result.progress.processed_steps == 3
        assert result.fees == 0.0
        assert result.initial_equity == 100000.0
        assert result.final_equity == 100000.0  # no trade matches in neutral predict fn
        assert len(result.determinism_hash) == 64
    finally:
        os.close(fd)
        os.remove(path)
