import os
import glob
import ast
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import pytest

from backend.decision.exceptions import DecisionError, InvalidPolicyError, ContextStoreError
from backend.decision.models import MLSignal, IntelligenceSnapshot, FusionInput, FusionResult, TradeProposal
from backend.decision.policy import FusionPolicy
from backend.decision.intelligence_context import IntelligenceContextStore
from backend.decision.fusion import DecisionFusionEngine
from backend.decision.telemetry import DecisionTelemetrySink
from backend.intelligence.models import ReasoningResult


# Custom Mock Telemetry Sink for testing
class MockTelemetrySink(DecisionTelemetrySink):
    def __init__(self):
        self.records = []

    def record(
        self,
        result: FusionResult,
        proposal: Optional[TradeProposal],
        latency_ms: float,
        rejection_reason: Optional[str] = None,
    ) -> None:
        self.records.append((result, proposal, latency_ms, rejection_reason))


@pytest.fixture
def default_policy():
    return FusionPolicy(
        policy_version="v1.0",
        ml_weight=0.7,
        intelligence_weight=0.3,
        minimum_ml_confidence=0.5,
        minimum_fusion_confidence=0.5,
        minimum_agreement_score=-1.0,  # Allow all agreements by default
        allow_ml_only=True,
        reject_on_critical_drift=True,
        reject_on_intelligence_risk_flags=["extreme_volatility", "unreliable_data"],
        proposal_ttl_seconds=60.0,
    )


@pytest.fixture
def fusion_engine(default_policy):
    sink = MockTelemetrySink()
    return DecisionFusionEngine(policy=default_policy, telemetry_sink=sink)


@pytest.fixture
def fresh_result():
    return ReasoningResult(
        market_regime="ranging",
        directional_bias="bearish",
        confidence=0.6,
        risk_flags=["low_rsi_divergence"],
        reasoning_summary="Bearish outlook on indicator breakdowns.",
        evidence=["RSI < 40"],
        provider="ollama",
        model="qwen3.5:4b",
        latency_ms=12000.0,
        request_id="req-999",
        prompt_id="market_reasoning",
        prompt_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =====================================================================
# 1. ARCHITECTURE IMPORT ISOLATION TEST
# =====================================================================
def test_execution_import_isolation():
    """
    Ensure backend/decision package has zero dependencies on brokers,
    exchanges, execution modules, PaperExecutor, etc.
    """
    decision_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "backend", "decision")
    )
    files = glob.glob(os.path.join(decision_dir, "*.py"))
    assert len(files) > 0, "No files found in backend/decision package"

    forbidden_keywords = [
        "broker",
        "exchange",
        "execution",
        "paperexecutor",
        "orderexecutor",
    ]

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        mod_name = name.name.lower()
                        for kw in forbidden_keywords:
                            if kw in mod_name:
                                raise AssertionError(
                                    f"Forbidden import '{name.name}' detected in file: {os.path.basename(f)}"
                                )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mod_name = node.module.lower()
                        for kw in forbidden_keywords:
                            if kw in mod_name:
                                raise AssertionError(
                                    f"Forbidden import from '{node.module}' detected in file: {os.path.basename(f)}"
                                )


# =====================================================================
# 2. INTELLIGENCE CONTEXT STORE TESTS
# =====================================================================
def test_context_store_put_and_get():
    store = IntelligenceContextStore(default_ttl_seconds=10.0)
    result = ReasoningResult(
        market_regime="trending",
        directional_bias="bullish",
        confidence=0.8,
        risk_flags=[],
        reasoning_summary="Trends strongly.",
        evidence=["EMA cross"],
        provider="ollama",
        model="qwen3.5:4b",
        latency_ms=100.0,
        request_id="req-123",
        prompt_id="p-1",
        prompt_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    snapshot = store.put("BTCUSDT", "5m", result)
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.timeframe == "5m"
    assert snapshot.request_id == "req-123"

    retrieved = store.get("BTCUSDT", "5m")
    assert retrieved is not None
    assert retrieved.snapshot_id == snapshot.snapshot_id


def test_context_store_expired_ttl():
    # TTL set to 1 second
    store = IntelligenceContextStore(default_ttl_seconds=1.0)
    result = ReasoningResult(
        market_regime="trending",
        directional_bias="bullish",
        confidence=0.8,
        risk_flags=[],
        reasoning_summary="Trends",
        evidence=[],
        provider="ollama",
        model="qwen",
        latency_ms=100.0,
        request_id="req-123",
        prompt_id="p-1",
        prompt_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    store.put("BTCUSDT", "5m", result)

    # Access after 1.5 seconds simulated check time
    future_time = datetime.now(timezone.utc) + timedelta(seconds=2.0)
    retrieved = store.get("BTCUSDT", "5m", now=future_time)
    assert retrieved is None, "Expired intelligence should return None"


def test_context_store_bounded_memory():
    # Context store max size = 3
    store = IntelligenceContextStore(default_ttl_seconds=10.0, max_size=3)

    def make_res():
        return ReasoningResult(
            market_regime="trending",
            directional_bias="bullish",
            confidence=0.8,
            risk_flags=[],
            reasoning_summary="sum",
            evidence=[],
            provider="ollama",
            model="qwen",
            latency_ms=10.0,
            request_id="req",
            prompt_id="p",
            prompt_version="v",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Insert 4 different symbols
    store.put("SYM1", "5m", make_res())
    store.put("SYM2", "5m", make_res())
    store.put("SYM3", "5m", make_res())
    store.put("SYM4", "5m", make_res())

    # SYM1 should be evicted (oldest inserted)
    assert store.get("SYM1", "5m") is None
    assert store.get("SYM2", "5m") is not None
    assert store.get("SYM4", "5m") is not None


def test_context_store_invalidate_and_clear():
    store = IntelligenceContextStore(default_ttl_seconds=10.0)
    res = ReasoningResult(
        market_regime="trending",
        directional_bias="bullish",
        confidence=0.8,
        risk_flags=[],
        reasoning_summary="sum",
        evidence=[],
        provider="o",
        model="m",
        latency_ms=10.0,
        request_id="r",
        prompt_id="p",
        prompt_version="v",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    store.put("BTCUSDT", "5m", res)
    store.invalidate("BTCUSDT", "5m")
    assert store.get("BTCUSDT", "5m") is None

    store.put("BTCUSDT", "5m", res)
    store.clear()
    assert store.get("BTCUSDT", "5m") is None


# =====================================================================
# 3. FUSION POLICY VALIDATIONS
# =====================================================================
def test_invalid_policy_weight():
    with pytest.raises(InvalidPolicyError):
        FusionPolicy(
            policy_version="1",
            ml_weight=-0.1,
            intelligence_weight=1.0,
            minimum_ml_confidence=0.5,
            minimum_fusion_confidence=0.5,
            minimum_agreement_score=0.0,
            allow_ml_only=True,
            reject_on_critical_drift=True,
        )

    with pytest.raises(InvalidPolicyError):
        FusionPolicy(
            policy_version="1",
            ml_weight=0.0,
            intelligence_weight=0.0,
            minimum_ml_confidence=0.5,
            minimum_fusion_confidence=0.5,
            minimum_agreement_score=0.0,
            allow_ml_only=True,
            reject_on_critical_drift=True,
        )


def test_policy_weight_normalization():
    policy = FusionPolicy(
        policy_version="1",
        ml_weight=10.0,
        intelligence_weight=10.0,
        minimum_ml_confidence=0.5,
        minimum_fusion_confidence=0.5,
        minimum_agreement_score=0.0,
        allow_ml_only=True,
        reject_on_critical_drift=True,
    )
    # Weights should normalize to 0.5 each
    assert policy.ml_weight == 0.5
    assert policy.intelligence_weight == 0.5


# =====================================================================
# 4. DECISION ENGINE FUSION SCENARIOS
# =====================================================================
def test_valid_ml_fresh_aligned_intelligence(fusion_engine, fresh_result):
    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BEARISH",  # Both aligned bearish
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    store = IntelligenceContextStore()
    snapshot = store.put("BTCUSDT", "5m", fresh_result)  # Fresh result is Bearish

    res, proposal = fusion_engine.fuse(ml_sig, snapshot)

    assert res.direction == "BEARISH"
    assert res.agreement_score == 1.0
    assert res.intelligence_used is True
    assert proposal is not None
    assert proposal.direction == "BEARISH"
    assert proposal.confidence == (0.8 * 0.7) + (0.6 * 0.3)


def test_valid_ml_fresh_conflicting_intelligence(fusion_engine, fresh_result):
    # ML is Bullish, LLM (fresh_result) is Bearish
    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BULLISH",
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    store = IntelligenceContextStore()
    snapshot = store.put("BTCUSDT", "5m", fresh_result)

    # Set minimum agreement score to 0.0 (conflicts result in -1.0, which should reject)
    policy = FusionPolicy(
        policy_version="v1.0",
        ml_weight=0.7,
        intelligence_weight=0.3,
        minimum_ml_confidence=0.5,
        minimum_fusion_confidence=0.5,
        minimum_agreement_score=0.0,
        allow_ml_only=True,
        reject_on_critical_drift=True,
    )
    engine = DecisionFusionEngine(policy=policy)

    res, proposal = engine.fuse(ml_sig, snapshot)
    assert res.agreement_score == -1.0
    assert res.direction == "NEUTRAL"
    assert proposal is None, "Proposal should be rejected due to disagreement"


def test_ml_only_fallback_allowed(default_policy):
    # Stale context/None, allow_ml_only is True
    engine = DecisionFusionEngine(policy=default_policy)
    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BULLISH",
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    res, proposal = engine.fuse(ml_sig, None)
    assert res.direction == "BULLISH"
    assert res.intelligence_used is False
    assert proposal is not None
    assert proposal.confidence == 0.8


def test_ml_only_fallback_rejected(default_policy):
    # Stale context/None, allow_ml_only is False -> Reject
    policy = FusionPolicy(
        policy_version="v1.0",
        ml_weight=0.7,
        intelligence_weight=0.3,
        minimum_ml_confidence=0.5,
        minimum_fusion_confidence=0.5,
        minimum_agreement_score=-1.0,
        allow_ml_only=False,
        reject_on_critical_drift=True,
    )
    engine = DecisionFusionEngine(policy=policy)

    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BULLISH",
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    res, proposal = engine.fuse(ml_sig, None)
    assert proposal is None
    assert "Intelligence Context is unavailable or stale" in res.metadata.get(
        "rejection_reason", ""
    )


def test_critical_ml_drift_rejection(fusion_engine):
    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BULLISH",
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="critical",  # CRITICAL
    )

    res, proposal = fusion_engine.fuse(ml_sig, None)
    assert proposal is None
    assert "Critical ML model drift detected" in res.metadata.get(
        "rejection_reason", ""
    )


def test_intel_risk_flags_rejection(fusion_engine, fresh_result):
    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BEARISH",
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    store = IntelligenceContextStore()
    # Add a critical risk flag that policy rejects: "extreme_volatility"
    fresh_result.risk_flags.append("extreme_volatility")
    snapshot = store.put("BTCUSDT", "5m", fresh_result)

    res, proposal = fusion_engine.fuse(ml_sig, snapshot)
    assert proposal is None
    assert "critical intelligence risk flags" in res.metadata.get(
        "rejection_reason", ""
    )


def test_invalid_directional_values(fusion_engine):
    ml_sig_bad = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="UPWARD_TRENDY",  # Invalid
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    with pytest.raises(DecisionError) as exc_info:
        fusion_engine.fuse(ml_sig_bad, None)
    assert "Invalid directional value detected" in str(exc_info.value)


def test_repeated_fusion_determinism(fusion_engine, fresh_result):
    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BEARISH",
        confidence=0.85,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )
    store = IntelligenceContextStore()
    snapshot = store.put("BTCUSDT", "5m", fresh_result)

    res1, prop1 = fusion_engine.fuse(ml_sig, snapshot)
    res2, prop2 = fusion_engine.fuse(ml_sig, snapshot)

    assert res1.fusion_score == res2.fusion_score
    assert res1.direction == res2.direction
    assert prop1.fusion_score == prop2.fusion_score
    assert prop1.proposal_id != prop2.proposal_id  # UUIDs are unique per call


def test_agreement_score_matrix(fusion_engine):
    # Test all agreement combinations
    assert fusion_engine.calculate_agreement("BULLISH", "BULLISH") == 1.0
    assert fusion_engine.calculate_agreement("BEARISH", "BEARISH") == 1.0
    assert fusion_engine.calculate_agreement("NEUTRAL", "NEUTRAL") == 1.0

    assert fusion_engine.calculate_agreement("BULLISH", "BEARISH") == -1.0
    assert fusion_engine.calculate_agreement("BEARISH", "BULLISH") == -1.0

    assert fusion_engine.calculate_agreement("NEUTRAL", "BULLISH") == 0.0
    assert fusion_engine.calculate_agreement("BEARISH", "NEUTRAL") == 0.0


def test_telemetry_and_latency_measurement(default_policy):
    sink = MockTelemetrySink()
    engine = DecisionFusionEngine(policy=default_policy, telemetry_sink=sink)

    ml_sig = MLSignal(
        model_version="MLv4",
        symbol="BTCUSDT",
        timeframe="5m",
        prediction=1.0,
        direction="BULLISH",
        confidence=0.8,
        calibrated=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        drift_status="normal",
    )

    t_start = time.perf_counter()
    res, prop = engine.fuse(ml_sig, None)
    t_duration = (time.perf_counter() - t_start) * 1000.0

    assert len(sink.records) == 1
    logged_res, logged_prop, logged_lat, _ = sink.records[0]

    assert prop is not None
    assert logged_prop is not None
    assert logged_res.fusion_id == res.fusion_id
    assert logged_prop.proposal_id == prop.proposal_id
    # Blockless engine execution should be virtually instant (<10 milliseconds)
    assert logged_lat < 10.0
