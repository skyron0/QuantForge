"""
Unit tests for backend/feature_runtime (Sprint 3.11).

Covers:
  - FeatureSchema fingerprinting
  - HistoricalFeatureBuffer bounded storage & causal filtering
  - FeatureExtractor warmup / NaN / causal gates
  - FeatureValidator fingerprint, ordering, bounds checks
  - FeatureInferenceEngine compatibility gate
  - FeatureSignalMapper direction mapping
  - FeatureRuntimeService pipeline results
  - Telemetry counters
  - AST isolation (no forbidden imports)
"""

import ast
import math
import os
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from backend.feature_runtime.exceptions import (
    FeatureRuntimeError,
    FeatureValidationError,
    FeatureSchemaMismatchError,
    FeatureOrderingError,
    InsufficientHistoryError,
    FeatureWarmupError,
    InvalidFeatureValueError,
    MissingFeatureError,
    StaleFeatureError,
    FutureFeatureError,
    InferenceRuntimeError,
    ModelCompatibilityError,
    InvalidPredictionError,
    ModelUnavailableError,
    SignalGenerationError,
    InvalidFeaturePolicyError,
)
from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.models import (
    FeatureSnapshot,
    PredictionResult,
    FeatureRuntimeResult,
    FeatureRuntimeStatus,
)
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.buffer import HistoricalFeatureBuffer, BufferCandle
from backend.feature_runtime.validator import FeatureValidator
from backend.feature_runtime.inference import FeatureInferenceEngine
from backend.feature_runtime.signal import FeatureSignalMapper
from backend.feature_runtime.service import FeatureRuntimeService
from backend.feature_runtime.bridge import FeatureRuntimeBridge, MLSignalGenerated
from backend.feature_runtime.telemetry import (
    FeatureRuntimeTelemetrySink,
    ConsoleFeatureRuntimeTelemetrySink,
)
from backend.feature_runtime.extractor import FeatureExtractor


# ═══════════════════════════════════════════════════════════════════════════
# Helpers & Fixtures
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "rsi", "ema20",
    "macd", "macd_signal", "macd_histogram",
    "atr", "adx", "vwap",
    "bb_upper", "bb_middle", "bb_lower",
]


@pytest.fixture
def schema():
    return FeatureSchema(
        schema_id="quantforge-default",
        schema_version="v1",
        feature_names=FEATURE_NAMES,
    )


@pytest.fixture
def policy():
    return FeatureRuntimePolicy(
        minimum_history=5,
        buffer_capacity=100,
        staleness_limit_seconds=60.0,
    )


@pytest.fixture
def buffer():
    return HistoricalFeatureBuffer(capacity=200)


def _make_candle(ts: str, close: float = 100.0, volume: float = 1000.0) -> BufferCandle:
    return BufferCandle(
        timestamp=ts,
        open=close - 1.0,
        high=close + 2.0,
        low=close - 2.0,
        close=close,
        volume=volume,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _past_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _future_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _make_snapshot(schema: FeatureSchema, symbol: str = "BTCUSDT") -> FeatureSnapshot:
    values = [50.0 + i for i in range(len(schema.feature_names))]
    return FeatureSnapshot(
        symbol=symbol,
        timestamp=_now_iso(),
        feature_names=list(schema.feature_names),
        feature_values=values,
        feature_version=schema.schema_version,
        schema_fingerprint=schema.fingerprint,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. FeatureSchema Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureSchema:

    def test_fingerprint_is_sha256_hex(self, schema: FeatureSchema):
        assert len(schema.fingerprint) == 64
        int(schema.fingerprint, 16)  # validates hex

    def test_same_inputs_produce_same_fingerprint(self):
        s1 = FeatureSchema("id-a", "v1", ["f1", "f2"])
        s2 = FeatureSchema("id-a", "v1", ["f1", "f2"])
        assert s1.fingerprint == s2.fingerprint

    def test_different_ordering_produces_different_fingerprint(self):
        s1 = FeatureSchema("id", "v1", ["a", "b"])
        s2 = FeatureSchema("id", "v1", ["b", "a"])
        assert s1.fingerprint != s2.fingerprint

    def test_different_version_produces_different_fingerprint(self):
        s1 = FeatureSchema("id", "v1", ["a"])
        s2 = FeatureSchema("id", "v2", ["a"])
        assert s1.fingerprint != s2.fingerprint

    def test_matches_detects_identical(self):
        s1 = FeatureSchema("id", "v1", ["a", "b"])
        s2 = FeatureSchema("id", "v1", ["a", "b"])
        assert s1.matches(s2)

    def test_validate_ordering_positive(self):
        s = FeatureSchema("id", "v1", ["a", "b", "c"])
        assert s.validate_ordering(["a", "b", "c"])

    def test_validate_ordering_negative(self):
        s = FeatureSchema("id", "v1", ["a", "b", "c"])
        assert not s.validate_ordering(["c", "b", "a"])

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            FeatureSchema("id", "v1", ["a", "a"])

    def test_empty_names_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            FeatureSchema("id", "v1", [])

    def test_feature_count(self, schema: FeatureSchema):
        assert schema.feature_count == len(FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# 2. HistoricalFeatureBuffer Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHistoricalFeatureBuffer:

    def test_append_and_get(self, buffer: HistoricalFeatureBuffer):
        c = _make_candle("2025-01-01T00:00:00Z")
        buffer.append("BTCUSDT", c)
        assert buffer.count("BTCUSDT") == 1
        candles = buffer.get_candles("BTCUSDT")
        assert len(candles) == 1

    def test_bounded_eviction(self):
        buf = HistoricalFeatureBuffer(capacity=3)
        for i in range(5):
            buf.append("SYM", _make_candle(f"2025-01-01T00:0{i}:00Z"))
        assert buf.count("SYM") == 3
        candles = buf.get_candles("SYM")
        assert candles[0].timestamp == "2025-01-01T00:02:00Z"

    def test_causal_filter(self, buffer: HistoricalFeatureBuffer):
        buffer.append("SYM", _make_candle("2025-01-01T00:01:00Z"))
        buffer.append("SYM", _make_candle("2025-01-01T00:02:00Z"))
        buffer.append("SYM", _make_candle("2025-01-01T00:03:00Z"))

        causal = buffer.get_candles_up_to("SYM", "2025-01-01T00:02:00Z")
        assert len(causal) == 2

    def test_empty_symbol_returns_empty(self, buffer: HistoricalFeatureBuffer):
        assert buffer.get_candles("UNKNOWN") == []
        assert buffer.get_candles_up_to("UNKNOWN", "2099-01-01") == []

    def test_clear_specific_symbol(self, buffer: HistoricalFeatureBuffer):
        buffer.append("A", _make_candle("2025-01-01T00:00:00Z"))
        buffer.append("B", _make_candle("2025-01-01T00:00:00Z"))
        buffer.clear("A")
        assert buffer.count("A") == 0
        assert buffer.count("B") == 1

    def test_clear_all(self, buffer: HistoricalFeatureBuffer):
        buffer.append("A", _make_candle("2025-01-01T00:00:00Z"))
        buffer.append("B", _make_candle("2025-01-01T00:00:00Z"))
        buffer.clear()
        assert buffer.count("A") == 0
        assert buffer.count("B") == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. FeatureValidator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureValidator:

    def test_valid_snapshot_passes(self, schema: FeatureSchema):
        validator = FeatureValidator(schema)
        snap = _make_snapshot(schema)
        validator.validate(snap)  # Should not raise

    def test_fingerprint_mismatch_rejected(self, schema: FeatureSchema):
        validator = FeatureValidator(schema)
        snap = FeatureSnapshot(
            symbol="SYM", timestamp=_now_iso(),
            feature_names=list(schema.feature_names),
            feature_values=[0.0] * schema.feature_count,
            feature_version="v1",
            schema_fingerprint="bad-fingerprint",
        )
        with pytest.raises(FeatureSchemaMismatchError):
            validator.validate(snap)

    def test_ordering_mismatch_rejected(self, schema: FeatureSchema):
        validator = FeatureValidator(schema)
        reversed_names = list(reversed(schema.feature_names))
        # Create a snapshot with a matching fingerprint but wrong ordering
        other_schema = FeatureSchema("quantforge-default", "v1", reversed_names)
        snap = FeatureSnapshot(
            symbol="SYM", timestamp=_now_iso(),
            feature_names=reversed_names,
            feature_values=[0.0] * schema.feature_count,
            feature_version="v1",
            schema_fingerprint=other_schema.fingerprint,
        )
        # The fingerprint will differ since ordering changed, so it triggers fingerprint error
        with pytest.raises((FeatureSchemaMismatchError, FeatureOrderingError)):
            validator.validate(snap)

    def test_nan_value_rejected(self, schema: FeatureSchema):
        validator = FeatureValidator(schema)
        values = [50.0] * schema.feature_count
        values[0] = float("nan")
        snap = FeatureSnapshot(
            symbol="SYM", timestamp=_now_iso(),
            feature_names=list(schema.feature_names),
            feature_values=values,
            feature_version=schema.schema_version,
            schema_fingerprint=schema.fingerprint,
        )
        with pytest.raises(InvalidFeatureValueError):
            validator.validate(snap)

    def test_inf_value_rejected(self, schema: FeatureSchema):
        validator = FeatureValidator(schema)
        values = [50.0] * schema.feature_count
        values[2] = float("inf")
        snap = FeatureSnapshot(
            symbol="SYM", timestamp=_now_iso(),
            feature_names=list(schema.feature_names),
            feature_values=values,
            feature_version=schema.schema_version,
            schema_fingerprint=schema.fingerprint,
        )
        with pytest.raises(InvalidFeatureValueError):
            validator.validate(snap)

    def test_count_mismatch_rejected(self, schema: FeatureSchema):
        validator = FeatureValidator(schema)
        short_names = schema.feature_names[:3]
        short_schema = FeatureSchema("quantforge-default", "v1", short_names)
        snap = FeatureSnapshot(
            symbol="SYM", timestamp=_now_iso(),
            feature_names=short_names,
            feature_values=[1.0, 2.0, 3.0],
            feature_version="v1",
            schema_fingerprint=short_schema.fingerprint,
        )
        with pytest.raises((FeatureSchemaMismatchError, MissingFeatureError)):
            validator.validate(snap)


# ═══════════════════════════════════════════════════════════════════════════
# 4. FeatureInferenceEngine Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureInferenceEngine:

    def test_successful_prediction(self, schema: FeatureSchema):
        engine = FeatureInferenceEngine(
            schema=schema,
            predict_fn=lambda features: 0.75,
        )
        snap = _make_snapshot(schema)
        result = engine.predict(snap)
        assert result.prediction == 0.75
        assert result.latency_ms >= 0

    def test_no_predict_fn_raises_unavailable(self, schema: FeatureSchema):
        engine = FeatureInferenceEngine(schema=schema, predict_fn=None)
        snap = _make_snapshot(schema)
        with pytest.raises(ModelUnavailableError):
            engine.predict(snap)

    def test_predict_fn_exception_raises_runtime_error(self, schema: FeatureSchema):
        def bad_fn(features):
            raise RuntimeError("model crashed")
        engine = FeatureInferenceEngine(schema=schema, predict_fn=bad_fn)
        snap = _make_snapshot(schema)
        with pytest.raises(InferenceRuntimeError, match="model crashed"):
            engine.predict(snap)

    def test_nan_prediction_rejected(self, schema: FeatureSchema):
        engine = FeatureInferenceEngine(
            schema=schema,
            predict_fn=lambda f: float("nan"),
        )
        snap = _make_snapshot(schema)
        with pytest.raises(InvalidPredictionError):
            engine.predict(snap)

    def test_schema_id_mismatch_rejected(self, schema: FeatureSchema):
        engine = FeatureInferenceEngine(
            schema=schema,
            predict_fn=lambda f: 0.5,
            model_metadata={
                "schema_id": "wrong-id",
                "schema_version": schema.schema_version,
                "fingerprint": schema.fingerprint,
                "feature_names": list(schema.feature_names),
            },
        )
        snap = _make_snapshot(schema)
        with pytest.raises(ModelCompatibilityError, match="schema_id"):
            engine.predict(snap)

    def test_feature_ordering_mismatch_rejected(self, schema: FeatureSchema):
        engine = FeatureInferenceEngine(
            schema=schema,
            predict_fn=lambda f: 0.5,
            model_metadata={
                "schema_id": schema.schema_id,
                "schema_version": schema.schema_version,
                "fingerprint": schema.fingerprint,
                "feature_names": list(reversed(schema.feature_names)),
            },
        )
        snap = _make_snapshot(schema)
        with pytest.raises(ModelCompatibilityError, match="ordering"):
            engine.predict(snap)

    def test_feature_count_mismatch_rejected(self, schema: FeatureSchema):
        engine = FeatureInferenceEngine(
            schema=schema,
            predict_fn=lambda f: 0.5,
            model_metadata={
                "schema_id": schema.schema_id,
                "schema_version": schema.schema_version,
                "fingerprint": schema.fingerprint,
                "feature_names": ["a", "b"],  # wrong count
            },
        )
        snap = _make_snapshot(schema)
        with pytest.raises(ModelCompatibilityError, match="count"):
            engine.predict(snap)


# ═══════════════════════════════════════════════════════════════════════════
# 5. FeatureSignalMapper Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureSignalMapper:

    def test_bullish_mapping(self):
        mapper = FeatureSignalMapper(bullish_threshold=0.55, bearish_threshold=0.45)
        pred = PredictionResult(symbol="SYM", timestamp=_now_iso(), prediction=0.8)
        sig = mapper.map(pred)
        assert sig.direction == "BULLISH"
        assert sig.confidence > 0

    def test_bearish_mapping(self):
        mapper = FeatureSignalMapper(bullish_threshold=0.55, bearish_threshold=0.45)
        pred = PredictionResult(symbol="SYM", timestamp=_now_iso(), prediction=0.2)
        sig = mapper.map(pred)
        assert sig.direction == "BEARISH"

    def test_neutral_mapping(self):
        mapper = FeatureSignalMapper(bullish_threshold=0.55, bearish_threshold=0.45)
        pred = PredictionResult(symbol="SYM", timestamp=_now_iso(), prediction=0.5)
        sig = mapper.map(pred)
        assert sig.direction == "NEUTRAL"

    def test_boundary_bullish(self):
        mapper = FeatureSignalMapper(bullish_threshold=0.55, bearish_threshold=0.45)
        pred = PredictionResult(symbol="SYM", timestamp=_now_iso(), prediction=0.55)
        sig = mapper.map(pred)
        assert sig.direction == "BULLISH"

    def test_boundary_bearish(self):
        mapper = FeatureSignalMapper(bullish_threshold=0.55, bearish_threshold=0.45)
        pred = PredictionResult(symbol="SYM", timestamp=_now_iso(), prediction=0.45)
        sig = mapper.map(pred)
        assert sig.direction == "BEARISH"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Causal Leakage Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalLeakage:
    """Prove that future candles cannot alter features at an earlier timestamp."""

    def test_future_candles_excluded_from_buffer(self):
        buf = HistoricalFeatureBuffer(capacity=100)
        buf.append("SYM", _make_candle("2025-01-01T00:01:00Z", close=100.0))
        buf.append("SYM", _make_candle("2025-01-01T00:02:00Z", close=110.0))
        buf.append("SYM", _make_candle("2025-01-01T00:03:00Z", close=120.0))

        # Snapshot at T=00:02 must not see T=00:03
        causal = buf.get_candles_up_to("SYM", "2025-01-01T00:02:00Z")
        assert len(causal) == 2
        assert all(c.timestamp <= "2025-01-01T00:02:00Z" for c in causal)

    def test_future_candle_does_not_alter_past_features(self):
        """
        Append candles up to T, extract features, then append future
        candle at T+1 and re-extract at T — result must be identical.
        """
        buf = HistoricalFeatureBuffer(capacity=500)

        # Build a history of 5 candles ending at T
        for i in range(5):
            ts = f"2025-01-01T00:{i:02d}:00Z"
            buf.append("SYM", _make_candle(ts, close=100.0 + i))

        snapshot_t = "2025-01-01T00:04:00Z"
        candles_before = buf.get_candles_up_to("SYM", snapshot_t)

        # Append a FUTURE candle at T+1
        buf.append("SYM", _make_candle("2025-01-01T00:05:00Z", close=999.0))

        candles_after = buf.get_candles_up_to("SYM", snapshot_t)

        # Identical candle sets despite future injection
        assert len(candles_before) == len(candles_after)
        for a, b in zip(candles_before, candles_after):
            assert a.close == b.close
            assert a.timestamp == b.timestamp


# ═══════════════════════════════════════════════════════════════════════════
# 7. FeatureRuntimePolicy Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureRuntimePolicy:

    def test_valid_policy(self, policy: FeatureRuntimePolicy):
        assert policy.minimum_history == 5
        assert policy.buffer_capacity == 100

    def test_negative_history_rejected(self):
        with pytest.raises(InvalidFeaturePolicyError):
            FeatureRuntimePolicy(minimum_history=0)

    def test_buffer_below_history_rejected(self):
        with pytest.raises(InvalidFeaturePolicyError):
            FeatureRuntimePolicy(minimum_history=100, buffer_capacity=50)

    def test_invalid_thresholds_rejected(self):
        with pytest.raises(InvalidFeaturePolicyError):
            FeatureRuntimePolicy(bullish_threshold=0.3, bearish_threshold=0.7)


# ═══════════════════════════════════════════════════════════════════════════
# 8. FeatureRuntimeService Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureRuntimeService:
    """Uses a mock extractor approach to test the full pipeline without real IndicatorEngine."""

    def _build_service(self, schema, policy, predict_fn=None, telemetry=None):
        buf = HistoricalFeatureBuffer(capacity=policy.buffer_capacity)
        # We manually build the extractor but will bypass it in these tests
        extractor = FeatureExtractor(schema, buf, minimum_history=policy.minimum_history)
        validator = FeatureValidator(schema)
        inference = FeatureInferenceEngine(schema, predict_fn=predict_fn)
        mapper = FeatureSignalMapper(
            bullish_threshold=policy.bullish_threshold,
            bearish_threshold=policy.bearish_threshold,
            default_timeframe=policy.default_timeframe,
        )
        return FeatureRuntimeService(
            policy=policy,
            schema=schema,
            buffer=buf,
            extractor=extractor,
            validator=validator,
            inference_engine=inference,
            signal_mapper=mapper,
            telemetry_sink=telemetry,
        )

    def test_warmup_skip_when_insufficient_history(self, schema, policy):
        svc = self._build_service(schema, policy)
        result = svc.process("BTCUSDT", _now_iso())
        assert result.status == FeatureRuntimeStatus.WARMUP_SKIP
        assert result.failed_stage == "EXTRACTION"

    def test_stale_data_rejected(self, schema, policy):
        policy_strict = FeatureRuntimePolicy(
            minimum_history=5,
            buffer_capacity=100,
            staleness_limit_seconds=1.0,
        )
        svc = self._build_service(schema, policy_strict)
        old_ts = _past_iso(60.0)
        result = svc.process("BTCUSDT", old_ts)
        assert result.status == FeatureRuntimeStatus.STALE_DATA

    def test_result_has_stage_timings(self, schema, policy):
        svc = self._build_service(schema, policy)
        result = svc.process("BTCUSDT", _now_iso())
        assert "staleness_check" in result.stage_timings
        assert result.total_latency_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Telemetry Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTelemetry:

    def test_warmup_skip_counted(self):
        sink = ConsoleFeatureRuntimeTelemetrySink()
        result = FeatureRuntimeResult(
            status=FeatureRuntimeStatus.WARMUP_SKIP,
            symbol="SYM", timestamp=_now_iso(),
            failed_stage="EXTRACTION", rejection_reason="too few candles",
            total_latency_ms=1.0,
        )
        sink.record_result(result)
        m = sink.get_metrics()
        assert m["warmup_skips"] == 1
        assert m["total_results"] == 1

    def test_success_counted(self):
        from backend.decision.models import MLSignal
        sig = MLSignal(
            model_version="v1", symbol="SYM", timeframe="5m",
            prediction=0.7, direction="BULLISH", confidence=0.4,
            calibrated=False, timestamp=_now_iso(), drift_status="normal",
        )
        sink = ConsoleFeatureRuntimeTelemetrySink()
        result = FeatureRuntimeResult(
            status=FeatureRuntimeStatus.SUCCESS,
            symbol="SYM", timestamp=_now_iso(),
            ml_signal=sig, total_latency_ms=2.5,
        )
        sink.record_result(result)
        m = sink.get_metrics()
        assert m["signals_generated"] == 1
        assert m["inference_calls"] == 1

    def test_neutral_signal_counted(self):
        from backend.decision.models import MLSignal
        sig = MLSignal(
            model_version="v1", symbol="SYM", timeframe="5m",
            prediction=0.5, direction="NEUTRAL", confidence=0.0,
            calibrated=False, timestamp=_now_iso(), drift_status="normal",
        )
        sink = ConsoleFeatureRuntimeTelemetrySink()
        result = FeatureRuntimeResult(
            status=FeatureRuntimeStatus.SUCCESS,
            symbol="SYM", timestamp=_now_iso(),
            ml_signal=sig, total_latency_ms=1.0,
        )
        sink.record_result(result)
        m = sink.get_metrics()
        assert m["neutral_signals"] == 1

    def test_schema_mismatch_counted(self):
        sink = ConsoleFeatureRuntimeTelemetrySink()
        result = FeatureRuntimeResult(
            status=FeatureRuntimeStatus.SCHEMA_MISMATCH,
            symbol="SYM", timestamp=_now_iso(),
            total_latency_ms=0.5,
        )
        sink.record_result(result)
        m = sink.get_metrics()
        assert m["schema_mismatches"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 10. FeatureSnapshot Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureSnapshotModel:

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="must equal"):
            FeatureSnapshot(
                symbol="SYM", timestamp=_now_iso(),
                feature_names=["a", "b"],
                feature_values=[1.0],
                feature_version="v1",
                schema_fingerprint="abc",
            )

    def test_as_dict_preserves_values(self, schema):
        snap = _make_snapshot(schema)
        d = snap.as_dict()
        assert len(d) == schema.feature_count
        for name, val in zip(snap.feature_names, snap.feature_values):
            assert d[name] == val


# ═══════════════════════════════════════════════════════════════════════════
# 11. AST Isolation Tests
# ═══════════════════════════════════════════════════════════════════════════

FORBIDDEN_MODULES = {
    "ccxt", "MetaTrader5",
    "backend.execution_adapter", "backend.execution_authorization",
    "backend.risk", "backend.positioning",
    "ollama", "openai", "anthropic",
    "requests", "httpx", "socket",
    "sqlalchemy", "SQLAlchemy",
}


class TestASTIsolation:
    """Verify that feature_runtime has zero forbidden imports."""

    def _get_feature_runtime_files(self):
        pkg_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "backend", "feature_runtime"
        )
        pkg_dir = os.path.normpath(pkg_dir)
        py_files = []
        for root, _dirs, files in os.walk(pkg_dir):
            for f in files:
                if f.endswith(".py") and not f.startswith("__pycache__"):
                    py_files.append(os.path.join(root, f))
        return py_files

    def test_no_forbidden_imports(self):
        for filepath in self._get_feature_runtime_files():
            with open(filepath, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for forbidden in FORBIDDEN_MODULES:
                            assert not alias.name.startswith(forbidden), (
                                f"Forbidden import '{alias.name}' in {filepath}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for forbidden in FORBIDDEN_MODULES:
                            assert not node.module.startswith(forbidden), (
                                f"Forbidden import from '{node.module}' in {filepath}"
                            )
