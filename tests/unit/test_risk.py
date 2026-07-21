import os
import ast
from datetime import datetime, timezone, timedelta
import pytest

from backend.decision.models import TradeProposal
from backend.risk.exceptions import (
    RiskError,
    InvalidRiskPolicyError,
    RiskContextError,
    RiskValidationError,
)
from backend.risk.models import (
    RiskContext,
    RiskAuthorizationStatus,
    RiskAuthorizationResult,
)
from backend.risk.policy import RiskPolicy
from backend.risk.guard import RiskGuardEngine
from backend.risk.telemetry import RiskTelemetrySink


# Mock Telemetry Sink for verification
class MockRiskTelemetrySink(RiskTelemetrySink):
    def __init__(self):
        self.calls = []

    def record(self, result: RiskAuthorizationResult, context: RiskContext, latency_ms: float) -> None:
        self.calls.append((result, context, latency_ms))


@pytest.fixture
def base_policy() -> RiskPolicy:
    return RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.60,
        maximum_proposal_age_seconds=10.0,
        maximum_daily_loss_fraction=0.03,  # 3% daily loss limit
        maximum_drawdown_fraction=0.05,   # 5% drawdown limit
        maximum_portfolio_exposure_fraction=0.30,  # 30% limit
        maximum_symbol_exposure_fraction=0.10,  # 10% limit
        maximum_leverage=3.0,
        maximum_open_positions=3,
        maximum_symbol_open_positions=1,
        maximum_consecutive_losses=3,
        reject_on_critical_volatility=True,
        reject_on_critical_liquidity=True,
        reject_on_critical_drift=True,
        base_risk_fraction=0.02,  # 2% standard risk
        maximum_risk_fraction=0.05,
        minimum_risk_fraction=0.005,
        blocking_risk_flags={"highly_unreliable", "illegal_arbitrage"},
        risk_reducing_flags={"medium_divergence": 0.50, "minor_spread": 0.80},
        informational_risk_flags={"low_volume", "weekend_trade"},
        volatility_adjustments={"HIGH": 0.50, "NORMAL": 1.0, "LOW": 1.0},
    )


@pytest.fixture
def base_proposal() -> TradeProposal:
    now_str = datetime.now(timezone.utc).isoformat()
    expires_str = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
    return TradeProposal(
        proposal_id="proposal-abc-123",
        symbol="BTC/USDT",
        direction="BULLISH",
        confidence=0.85,
        fusion_score=0.85,
        source_model_version="ml-v2",
        fusion_policy_version="fusion-1.0",
        reasoning_request_id=None,
        created_at=now_str,
        expires_at=expires_str,
        risk_flags=[],
        metadata={},
    )


@pytest.fixture
def base_context() -> RiskContext:
    return RiskContext(
        symbol="BTC/USDT",
        timestamp=datetime.now(timezone.utc).isoformat(),
        equity=10000.0,
        available_balance=5000.0,
        daily_realized_pnl=0.0,
        daily_unrealized_pnl=0.0,
        current_drawdown_pct=0.01,  # 1%
        portfolio_exposure_pct=0.15,  # 15%
        symbol_exposure_pct=0.05,   # 5%
        current_leverage=1.5,
        open_positions_count=1,
        symbol_open_positions_count=0,
        volatility_state="NORMAL",
        consecutive_losses=0,
        market_liquidity_state="NORMAL",
    )


# 1. Valid proposal APPROVED
def test_valid_proposal_approved(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)
    result = engine.evaluate(base_proposal, base_context)

    assert result.status == RiskAuthorizationStatus.APPROVED
    assert result.authorized_risk_fraction == base_policy.base_risk_fraction
    assert len(result.rejection_reasons) == 0
    assert result.original_confidence == base_proposal.confidence
    assert result.effective_confidence == base_proposal.confidence
    assert result.symbol == base_proposal.symbol
    assert result.direction == base_proposal.direction
    assert result.proposal_id == base_proposal.proposal_id
    assert result.source_model_version == base_proposal.source_model_version
    assert result.fusion_policy_version == base_proposal.fusion_policy_version
    assert result.evaluated_at is not None
    assert result.latency_ms > 0.0


# 2. Expired proposal REJECTED
def test_expired_proposal_rejected(base_policy, base_proposal, base_context):
    past_str = (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat()
    expired_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=past_str,
        expires_at=past_str,  # expires_at has passed
        risk_flags=base_proposal.risk_flags,
        metadata=base_proposal.metadata,
    )

    engine = RiskGuardEngine(policy=base_policy)
    result = engine.evaluate(expired_proposal, base_context)

    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "expired" in "".join(result.rejection_reasons).lower()
    assert "PROPOSAL_FRESHNESS_GATE" in result.triggered_rules
    assert result.authorized_risk_fraction == 0.0
    assert result.effective_confidence == 0.0


# 3. Future-dated or stale proposal REJECTED
def test_stale_and_future_proposal_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)
    
    # Too old:
    stale_str = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
    future_expire = (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()
    stale_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=stale_str,
        expires_at=future_expire,
        risk_flags=base_proposal.risk_flags,
        metadata=base_proposal.metadata,
    )
    result_stale = engine.evaluate(stale_proposal, base_context)
    assert result_stale.status == RiskAuthorizationStatus.REJECTED
    assert "too stale" in "".join(result_stale.rejection_reasons)
    
    # Future dated:
    future_str = (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()
    future_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=future_str,
        expires_at=future_str,
        risk_flags=base_proposal.risk_flags,
        metadata=base_proposal.metadata,
    )
    result_future = engine.evaluate(future_proposal, base_context)
    assert result_future.status == RiskAuthorizationStatus.REJECTED
    assert "future" in "".join(result_future.rejection_reasons)


# 4. Low/Invalid confidence REJECTED
def test_confidence_validation(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    # Under minimum
    low_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=0.55,  # under 0.60
        fusion_score=0.55,
        source_model_version="ml-v2",
        fusion_policy_version="fusion-1.0",
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=[],
        metadata={},
    )
    result_low = engine.evaluate(low_proposal, base_context)
    assert result_low.status == RiskAuthorizationStatus.REJECTED
    assert "under minimum" in "".join(result_low.rejection_reasons)

    # Invalid confidence (NaN/bounds)
    invalid_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=1.2,  # out of [0, 1]
        fusion_score=1.2,
        source_model_version="ml-v2",
        fusion_policy_version="fusion-1.0",
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=[],
        metadata={},
    )
    result_invalid = engine.evaluate(invalid_proposal, base_context)
    assert result_invalid.status == RiskAuthorizationStatus.REJECTED
    assert "INTEGRITY" in "".join(result_invalid.triggered_rules)


# 5. Daily Loss Gate verification (realized + unrealized loss limit)
def test_daily_loss_gate_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)
    
    # 3% of 10000 equity is 300. Let's make loss equal to 300.
    loss_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=-200.0,
        daily_unrealized_pnl=-100.0, # Total loss = 300.0 (3.0%)
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    
    result = engine.evaluate(base_proposal, loss_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "daily loss limit" in "".join(result.rejection_reasons).lower()
    assert "DAILY_LOSS_GATE" in result.triggered_rules


# 6. Drawdown Gate verification
def test_drawdown_gate_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    drawdown_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=0.06,  # 6% (limit is 5%)
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )

    result = engine.evaluate(base_proposal, drawdown_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "drawdown limit" in "".join(result.rejection_reasons).lower()
    assert "DRAWDOWN_GATE" in result.triggered_rules


# 7. Portfolio & Symbol exposure gates
def test_exposure_gates_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    # Portfolio exposure limit is 30%. Let's violate it
    port_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=0.35,  # 35%
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    result_port = engine.evaluate(base_proposal, port_context)
    assert result_port.status == RiskAuthorizationStatus.REJECTED
    assert "portfolio exposure" in "".join(result_port.rejection_reasons).lower()

    # Symbol exposure limit is 10%
    symbol_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=0.12,  # 12%
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    result_symbol = engine.evaluate(base_proposal, symbol_context)
    assert result_symbol.status == RiskAuthorizationStatus.REJECTED
    assert "symbol exposure" in "".join(result_symbol.rejection_reasons).lower()


# 8. Leverage Gate
def test_leverage_gate_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    lev_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=4.0,  # limit is 3.0
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )

    result = engine.evaluate(base_proposal, lev_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "leverage limit" in "".join(result.rejection_reasons).lower()


# 9. Open position limits
def test_open_position_limits_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    # General open positions limit is 3
    pos_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=3,  # limit is 3
        symbol_open_positions_count=0,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    result_pos = engine.evaluate(base_proposal, pos_context)
    assert result_pos.status == RiskAuthorizationStatus.REJECTED
    assert "maximum open positions" in "".join(result_pos.rejection_reasons).lower()

    # Symbol-specific positions limit is 1
    sym_pos_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=1,
        symbol_open_positions_count=1,  # limit is 1
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    result_sym_pos = engine.evaluate(base_proposal, sym_pos_context)
    assert result_sym_pos.status == RiskAuthorizationStatus.REJECTED
    assert "maximum open positions for symbol" in "".join(result_sym_pos.rejection_reasons).lower()


# 10. Consecutive loss circuit breaker
def test_consecutive_loss_gate(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    loss_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=3,  # limit is 3
        market_liquidity_state=base_context.market_liquidity_state,
    )

    result = engine.evaluate(base_proposal, loss_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "circuit breaker" in "".join(result.rejection_reasons).lower()


# 11. Volatility and Liquidity rejection gates
def test_volatility_and_liquidity_gates(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    # Volatility Crit
    vol_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state="CRITICAL",
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    result_vol = engine.evaluate(base_proposal, vol_context)
    assert result_vol.status == RiskAuthorizationStatus.REJECTED
    assert "critical market volatility" in "".join(result_vol.rejection_reasons).lower()

    # Liquidity Crit
    liq_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state=base_context.volatility_state,
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state="CRITICAL",
    )
    result_liq = engine.evaluate(base_proposal, liq_context)
    assert result_liq.status == RiskAuthorizationStatus.REJECTED
    assert "critical market liquidity" in "".join(result_liq.rejection_reasons).lower()


# 12. Critical ML Drift defense-in-depth rejection
def test_ml_drift_defense_in_depth(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    drift_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata={"drift_status": "critical"},
    )

    result = engine.evaluate(drift_proposal, base_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "drift detected" in "".join(result.rejection_reasons).lower()


# 13. Blocking Risk flags rejection
def test_blocking_risk_flags(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    blocked_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=["highly_unreliable"],  # blocking tag
        metadata=base_proposal.metadata,
    )

    result = engine.evaluate(blocked_proposal, base_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "blocking risk flags detected" in "".join(result.rejection_reasons).lower()


# 14. Unrecognized Safety-Sensitive Flags Rejection (Fail-Closed)
def test_unrecognized_safety_flags_rejected(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    unrecognized_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=["unknown_exploit_hazard"],  # not declared in policy
        metadata=base_proposal.metadata,
    )

    result = engine.evaluate(unrecognized_proposal, base_context)
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "unrecognized safety-sensitive flags" in "".join(result.rejection_reasons).lower()
    assert "UNRECOGNIZED_RISK_FLAGS" in result.triggered_rules


# 15. Risk Reducing flags and High Volatility adjustments
def test_risk_adjustments(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)

    # HIGH Volatility context (multiplier = 0.50)
    high_vol_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state="HIGH",
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    result = engine.evaluate(base_proposal, high_vol_context)
    assert result.status == RiskAuthorizationStatus.ADJUSTED
    # 0.02 base * 0.50 vol adjustment = 0.01
    assert result.authorized_risk_fraction == 0.01
    assert "VOLATILITY_ADJUSTMENT" in result.triggered_rules

    # Reducing flag (medium_divergence multiplier = 0.50)
    reducing_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=["medium_divergence"],
        metadata=base_proposal.metadata,
    )
    result_flag = engine.evaluate(reducing_proposal, base_context)
    assert result_flag.status == RiskAuthorizationStatus.ADJUSTED
    # 0.02 base * 0.50 flag adjustment = 0.01
    assert result_flag.authorized_risk_fraction == 0.01
    assert "RISK_REDUCING_FLAG_ADJUSTMENT" in result_flag.triggered_rules

    # Combined high vol + reducing flag: 0.02 * 0.50 * 0.50 = 0.005 (exactly equal to minimum flooring)
    result_combined = engine.evaluate(reducing_proposal, high_vol_context)
    assert result_combined.status == RiskAuthorizationStatus.ADJUSTED
    assert result_combined.authorized_risk_fraction == 0.005


# 16. Floor limit rejection and capping limit behavior
def test_floor_and_cap_limits(base_policy, base_proposal, base_context):
    # Cap test: policy maximum is 0.05. Let's make base risk fraction 0.05
    capped_policy = RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.60,
        maximum_proposal_age_seconds=10.0,
        maximum_daily_loss_fraction=0.03,
        maximum_drawdown_fraction=0.05,
        maximum_portfolio_exposure_fraction=0.30,
        maximum_symbol_exposure_fraction=0.10,
        maximum_leverage=3.0,
        maximum_open_positions=3,
        maximum_symbol_open_positions=1,
        maximum_consecutive_losses=3,
        reject_on_critical_volatility=True,
        reject_on_critical_liquidity=True,
        reject_on_critical_drift=True,
        base_risk_fraction=0.05,  # Equal to max
        maximum_risk_fraction=0.05,
        minimum_risk_fraction=0.01,
        blocking_risk_flags=set(),
        risk_reducing_flags=dict(),
        informational_risk_flags=set(),
        volatility_adjustments=dict(),
    )
    engine_capped = RiskGuardEngine(policy=capped_policy)
    result_cap = engine_capped.evaluate(base_proposal, base_context)
    # Status should be APPROVED (not adjusted since original is exactly max and no mult applied)
    assert result_cap.status == RiskAuthorizationStatus.APPROVED
    assert result_cap.authorized_risk_fraction == 0.05

    # Floor test: adjustments reduce to 0.004 (which is below minimum floor 0.005)
    # 0.02 * 0.50 (divergence) * 0.40 (vol) = 0.004 < 0.005 -> REJECTED
    floor_policy = RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.60,
        maximum_proposal_age_seconds=10.0,
        maximum_daily_loss_fraction=0.03,
        maximum_drawdown_fraction=0.05,
        maximum_portfolio_exposure_fraction=0.30,
        maximum_symbol_exposure_fraction=0.10,
        maximum_leverage=3.0,
        maximum_open_positions=3,
        maximum_symbol_open_positions=1,
        maximum_consecutive_losses=3,
        reject_on_critical_volatility=True,
        reject_on_critical_liquidity=True,
        reject_on_critical_drift=True,
        base_risk_fraction=0.02,
        maximum_risk_fraction=0.05,
        minimum_risk_fraction=0.005,
        blocking_risk_flags=set(),
        risk_reducing_flags={"medium_divergence": 0.50},
        informational_risk_flags=set(),
        volatility_adjustments={"HIGH": 0.40},
    )
    high_vol_context = RiskContext(
        symbol=base_context.symbol,
        timestamp=base_context.timestamp,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        daily_realized_pnl=base_context.daily_realized_pnl,
        daily_unrealized_pnl=base_context.daily_unrealized_pnl,
        current_drawdown_pct=base_context.current_drawdown_pct,
        portfolio_exposure_pct=base_context.portfolio_exposure_pct,
        symbol_exposure_pct=base_context.symbol_exposure_pct,
        current_leverage=base_context.current_leverage,
        open_positions_count=base_context.open_positions_count,
        symbol_open_positions_count=base_context.symbol_open_positions_count,
        volatility_state="HIGH",
        consecutive_losses=base_context.consecutive_losses,
        market_liquidity_state=base_context.market_liquidity_state,
    )
    reducing_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=0.85,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=None,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=["medium_divergence"],
        metadata=base_proposal.metadata,
    )
    engine_floor = RiskGuardEngine(policy=floor_policy)
    result_floor = engine_floor.evaluate(reducing_proposal, high_vol_context)
    assert result_floor.status == RiskAuthorizationStatus.REJECTED
    assert "fell below minimum" in "".join(result_floor.rejection_reasons)
    assert "MIN_RISK_FLOOR_REJECT" in result_floor.triggered_rules
    assert result_floor.authorized_risk_fraction == 0.0


# 17. Telemetry execution and latency logging verification
def test_risk_telemetry(base_policy, base_proposal, base_context):
    mock_telemetry = MockRiskTelemetrySink()
    engine = RiskGuardEngine(policy=base_policy, telemetry_sink=mock_telemetry)

    result = engine.evaluate(base_proposal, base_context)
    assert len(mock_telemetry.calls) == 1
    stored_result, stored_context, latency = mock_telemetry.calls[0]
    assert stored_result == result
    assert stored_context == base_context
    assert latency > 0.0


# 18. Invalid context inputs fail-closed (NaN checks)
def test_invalid_context_fail_closed(base_policy, base_proposal):
    with pytest.raises(RiskContextError):
        # NaN available_balance must reject creation
        RiskContext(
            symbol="BTC/USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            equity=10000.0,
            available_balance=float("nan"),
            daily_realized_pnl=0.0,
            daily_unrealized_pnl=0.0,
            current_drawdown_pct=0.01,
            portfolio_exposure_pct=0.15,
            symbol_exposure_pct=0.05,
            current_leverage=1.5,
            open_positions_count=1,
            symbol_open_positions_count=0,
            volatility_state="NORMAL",
            consecutive_losses=0,
        )

    # If somehow corrupted inputs reach the evaluation pipeline bypass post_init
    engine = RiskGuardEngine(policy=base_policy)
    # We pass malformed inputs to evaluate
    result = engine.evaluate(None, None)  # type: ignore
    assert result.status == RiskAuthorizationStatus.REJECTED
    assert "not an instance of tradeproposal" in "".join(result.rejection_reasons).lower()


# 19. Repeated evaluations have identical outputs
def test_repeated_evaluations_deterministic(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)
    result_1 = engine.evaluate(base_proposal, base_context)
    result_2 = engine.evaluate(base_proposal, base_context)

    # Identical except for random UUID authorization_id and evaluation timestamp discrepancies
    # We check core attributes match
    assert result_1.status == result_2.status
    assert result_1.authorized_risk_fraction == result_2.authorized_risk_fraction
    assert result_1.rejection_reasons == result_2.rejection_reasons
    assert result_1.effective_confidence == result_2.effective_confidence
    assert result_1.triggered_rules == result_2.triggered_rules


# 20. Policy validation checks (InvalidRiskPolicyError)
def test_policy_validations():
    # Negative leverage
    with pytest.raises(InvalidRiskPolicyError):
        RiskPolicy(
            policy_version="1.0.0",
            minimum_proposal_confidence=0.60,
            maximum_proposal_age_seconds=10.0,
            maximum_daily_loss_fraction=0.03,
            maximum_drawdown_fraction=0.05,
            maximum_portfolio_exposure_fraction=0.30,
            maximum_symbol_exposure_fraction=0.10,
            maximum_leverage=-1.5,  # negative leverage invalid
            maximum_open_positions=3,
            maximum_symbol_open_positions=1,
            maximum_consecutive_losses=3,
            reject_on_critical_volatility=True,
            reject_on_critical_liquidity=True,
            reject_on_critical_drift=True,
            base_risk_fraction=0.02,
            maximum_risk_fraction=0.05,
            minimum_risk_fraction=0.005,
        )

    # min_risk > max_risk
    with pytest.raises(InvalidRiskPolicyError):
        RiskPolicy(
            policy_version="1.0.0",
            minimum_proposal_confidence=0.60,
            maximum_proposal_age_seconds=10.0,
            maximum_daily_loss_fraction=0.03,
            maximum_drawdown_fraction=0.05,
            maximum_portfolio_exposure_fraction=0.30,
            maximum_symbol_exposure_fraction=0.10,
            maximum_leverage=3.0,
            maximum_open_positions=3,
            maximum_symbol_open_positions=1,
            maximum_consecutive_losses=3,
            reject_on_critical_volatility=True,
            reject_on_critical_liquidity=True,
            reject_on_critical_drift=True,
            base_risk_fraction=0.02,
            maximum_risk_fraction=0.01,  # max is 1%
            minimum_risk_fraction=0.02,  # min is 2%
        )


# 21. Static analysis: execution import isolation
def test_risk_package_execution_isolation():
    """Ensure backend/risk module doesn't import any execution-specific or broker modules."""
    forbidden_keywords = [
        "ccxt",
        "binance",
        "bybit",
        "backend.execution",
        "PaperExecutor",
        "LiveExecutor",
        "OrderExecutor",
        "brokers",
        "exchanges",
    ]

    risk_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "backend",
        "risk"
    )
    
    python_files = [
        os.path.join(risk_dir, f)
        for f in os.listdir(risk_dir)
        if f.endswith(".py")
    ]

    for filepath in python_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST code structure
        tree = ast.parse(content, filename=filepath)
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for keyword in forbidden_keywords:
                        assert keyword not in alias.name, (
                            f"Forbidden import '{alias.name}' detected in {filepath}."
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for keyword in forbidden_keywords:
                        assert keyword not in node.module, (
                            f"Forbidden import module '{node.module}' detected in {filepath}."
                        )
                    # Check imported aliases
                    for alias in node.names:
                        assert keyword not in alias.name, (
                            f"Forbidden import name '{alias.name}' from '{node.module}' in {filepath}."
                        )


# 22. Audit lineage is fully preserved
def test_audit_lineage_preserved(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)
    result = engine.evaluate(base_proposal, base_context)

    assert result.proposal_id == base_proposal.proposal_id
    assert result.source_model_version == base_proposal.source_model_version
    assert result.fusion_policy_version == base_proposal.fusion_policy_version
    assert result.policy_version == base_policy.policy_version
    assert result.proposal_created_at == base_proposal.created_at
    assert result.reasoning_request_id == base_proposal.reasoning_request_id


# 23. Verify execution-order isolation boundary (no quantity, no leverage mutation)
def test_execution_order_isolation_boundary(base_policy, base_proposal, base_context):
    engine = RiskGuardEngine(policy=base_policy)
    result = engine.evaluate(base_proposal, base_context)

    # RiskAuthorizationResult must not carry execution specific execution order states
    assert not hasattr(result, "quantity")
    assert not hasattr(result, "final_quantity")
    assert not hasattr(result, "order_type")
    assert not hasattr(result, "broker_order_id")
    
    # Engine should not modify leverage in context or propose brokerage leverage changes
    assert base_context.current_leverage == 1.5
    assert not hasattr(result, "target_leverage")

