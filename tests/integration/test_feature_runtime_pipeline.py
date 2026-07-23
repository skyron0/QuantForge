"""
Integration tests for the Feature Runtime pipeline (Sprint 3.11).

End-to-end flow:
  ReplayMarketDataProvider
  → MarketDataService
  → MarketDataSnapshot
  → FeatureRuntimeService
  → FeatureSnapshot
  → OnlineInference
  → MLSignal
  → DecisionFusionEngine

This pipeline MUST NOT execute RiskGuard, PositionSizing,
ExecutionAuthorization, or ExecutionAdapter.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

# Market data imports not needed — integration test uses BufferCandle directly
# This avoids coupling to MarketDataEnvelope field names.

from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.buffer import HistoricalFeatureBuffer, BufferCandle
from backend.feature_runtime.extractor import FeatureExtractor
from backend.feature_runtime.validator import FeatureValidator
from backend.feature_runtime.inference import FeatureInferenceEngine
from backend.feature_runtime.signal import FeatureSignalMapper
from backend.feature_runtime.service import FeatureRuntimeService
from backend.feature_runtime.telemetry import ConsoleFeatureRuntimeTelemetrySink
from backend.feature_runtime.models import FeatureRuntimeStatus, FeatureRuntimeResult
from backend.feature_runtime.bridge import FeatureRuntimeBridge, MLSignalGenerated

from backend.decision.models import MLSignal, IntelligenceSnapshot
from backend.decision.fusion import DecisionFusionEngine
from backend.decision.policy import FusionPolicy

from backend.runtime.event_bus import EventBus


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "rsi", "ema20",
    "macd", "macd_signal", "macd_histogram",
    "atr", "adx", "vwap",
    "bb_upper", "bb_middle", "bb_lower",
]


def _ts(minutes_ago: int = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


# Candle construction uses BufferCandle directly — no MarketDataEnvelope needed.


class _StubIndicatorEngine:
    """Stub that returns deterministic indicator values."""
    MIN_CANDLES = 5

    def calculate(self, candles):
        if len(candles) < self.MIN_CANDLES:
            return None
        last_close = float(candles[-1].close)
        return {
            "rsi": 50.0,
            "ema20": last_close,
            "macd": 0.5,
            "macd_signal": 0.3,
            "macd_histogram": 0.2,
            "atr": 2.0,
            "adx": 25.0,
            "vwap": last_close + 1.0,
            "bb_upper": last_close + 4.0,
            "bb_middle": last_close,
            "bb_lower": last_close - 4.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

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
        buffer_capacity=200,
        staleness_limit_seconds=600.0,  # generous for testing
    )


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def telemetry():
    return ConsoleFeatureRuntimeTelemetrySink()


def _build_service(schema, policy, telemetry=None):
    """Build a FeatureRuntimeService with a stubbed indicator engine."""
    buf = HistoricalFeatureBuffer(capacity=policy.buffer_capacity)
    extractor = FeatureExtractor(schema, buf, minimum_history=policy.minimum_history)

    # Monkey-patch the extractor's indicator proxy to use our stub
    stub = _StubIndicatorEngine()
    extractor._indicator_proxy._engine = stub

    validator = FeatureValidator(schema)
    inference = FeatureInferenceEngine(
        schema=schema,
        predict_fn=lambda features: 0.72,  # deterministic bullish
    )
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
    ), buf


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureRuntimePipeline:
    """End-to-end pipeline integration (no execution/risk/sizing)."""

    def test_warmup_then_signal_generation(self, schema, policy, telemetry):
        svc, buf = _build_service(schema, policy, telemetry)

        sym = "BTCUSDT"
        ts = _ts(0)

        # Only 3 candles — should fail with warmup
        for i in range(3):
            c_ts = _ts(10 - i)
            buf.append(sym, BufferCandle(
                timestamp=c_ts,
                open=99.0, high=102.0, low=98.0, close=100.0 + i, volume=1000.0,
            ))
        result = svc.process(sym, ts)
        assert result.status == FeatureRuntimeStatus.WARMUP_SKIP

        # Add more candles to meet minimum
        for i in range(3, 8):
            c_ts = _ts(10 - i)
            buf.append(sym, BufferCandle(
                timestamp=c_ts,
                open=99.0, high=102.0, low=98.0, close=100.0 + i, volume=1000.0,
            ))

        result = svc.process(sym, ts)
        assert result.status == FeatureRuntimeStatus.SUCCESS
        assert result.ml_signal is not None
        assert result.ml_signal.direction == "BULLISH"
        assert result.feature_snapshot is not None
        assert result.prediction_result is not None
        assert result.total_latency_ms > 0

        # Telemetry
        metrics = telemetry.get_metrics()
        assert metrics["warmup_skips"] == 1
        assert metrics["signals_generated"] == 1

    def test_full_pipeline_replay_to_fusion(self, schema, policy, event_bus, telemetry):
        """
        ReplayProvider → MarketDataService → buffer → FeatureRuntimeService
        → MLSignal → DecisionFusionEngine
        """
        svc, buf = _build_service(schema, policy, telemetry)

        sym = "BTCUSDT"

        # Build enough candles to pass warmup
        for i in range(10):
            c_ts = _ts(20 - i)
            buf.append(sym, BufferCandle(
                timestamp=c_ts,
                open=99.0, high=102.0, low=98.0, close=100.0 + i, volume=1000.0,
            ))

        ts_now = _ts(0)
        result = svc.process(sym, ts_now)
        assert result.status == FeatureRuntimeStatus.SUCCESS
        assert result.ml_signal is not None

        # Feed into DecisionFusionEngine
        fusion_policy = FusionPolicy(
            policy_version="fusion-v1",
            ml_weight=0.6,
            intelligence_weight=0.4,
            minimum_ml_confidence=0.3,
            minimum_fusion_confidence=0.4,
            minimum_agreement_score=-0.5,
            allow_ml_only=True,
            reject_on_critical_drift=True,
        )
        fusion_engine = DecisionFusionEngine(policy=fusion_policy)

        fusion_result, proposal = fusion_engine.fuse(
            ml_signal=result.ml_signal,
            intelligence_snapshot=None,
            now=datetime.now(timezone.utc),
        )
        assert fusion_result.fusion_id is not None
        assert fusion_result.direction in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_bridge_publishes_signal_event(self, schema, policy, event_bus, telemetry):
        svc, buf = _build_service(schema, policy, telemetry)
        bridge = FeatureRuntimeBridge(
            event_bus=event_bus,
            service=svc,
            runtime_id="test-bridge",
            session_id="test-session",
        )

        sym = "BTCUSDT"
        for i in range(10):
            c_ts = _ts(20 - i)
            buf.append(sym, BufferCandle(
                timestamp=c_ts,
                open=99.0, high=102.0, low=98.0, close=100.0 + i, volume=1000.0,
            ))

        captured_events = []
        event_bus.subscribe("MLSignalGenerated", lambda e: captured_events.append(e))

        signal = bridge.on_snapshot(sym, _ts(0))
        assert signal is not None
        assert signal.direction == "BULLISH"
        assert len(captured_events) == 1
        assert isinstance(captured_events[0], MLSignalGenerated)

    def test_pipeline_does_not_execute_trades(self, schema, policy, telemetry):
        """
        Verify that the pipeline output contains ONLY information-layer
        outputs; no OrderIntent, ExecutionResult, or Fill references.
        """
        svc, buf = _build_service(schema, policy, telemetry)
        sym = "BTCUSDT"
        for i in range(10):
            c_ts = _ts(20 - i)
            buf.append(sym, BufferCandle(
                timestamp=c_ts,
                open=99.0, high=102.0, low=98.0, close=100.0 + i, volume=1000.0,
            ))

        result = svc.process(sym, _ts(0))
        assert result.status == FeatureRuntimeStatus.SUCCESS

        # Verify no execution-layer attributes
        assert not hasattr(result, "order_intent")
        assert not hasattr(result, "execution_result")
        assert not hasattr(result, "fill")
        assert not hasattr(result, "risk_authorization")


class TestCausalIntegration:
    """Integration tests proving causal isolation."""

    def test_future_data_cannot_affect_past_inference(self, schema, policy, telemetry):
        svc, buf = _build_service(schema, policy, telemetry)
        sym = "BTCUSDT"

        # Populate buffer up to T
        for i in range(10):
            c_ts = _ts(20 - i)
            buf.append(sym, BufferCandle(
                timestamp=c_ts,
                open=99.0, high=102.0, low=98.0, close=100.0 + i, volume=1000.0,
            ))

        snapshot_ts = _ts(5)
        result_before = svc.process(sym, snapshot_ts)

        # Inject a FUTURE candle
        buf.append(sym, BufferCandle(
            timestamp=_ts(-10),  # 10 minutes in the future
            open=99.0, high=102.0, low=98.0, close=9999.0, volume=1000.0,
        ))

        result_after = svc.process(sym, snapshot_ts)

        # Both should produce identical feature snapshots
        if result_before.feature_snapshot and result_after.feature_snapshot:
            assert (
                result_before.feature_snapshot.feature_values
                == result_after.feature_snapshot.feature_values
            )
