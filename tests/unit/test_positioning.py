import os
import ast
import pytest
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from backend.decision.models import TradeProposal
from backend.risk.models import RiskAuthorizationResult, RiskAuthorizationStatus
from backend.positioning.exceptions import (
    PositionSizingError,
    PositionSizingValidationError,
    InvalidStopDistanceError,
    InsufficientCapitalError,
    PositionLimitError,
    AuthorizationError,
)
from backend.positioning.models import PositionSizingContext, PositionSizeResult
from backend.positioning.policy import PositionSizingPolicy
from backend.positioning.telemetry import PositionSizingTelemetrySink
from backend.positioning.sizing import PositionSizingEngine


class MockPositionSizingTelemetrySink(PositionSizingTelemetrySink):
    def __init__(self):
        self.calls = []

    def record(
        self,
        result: Optional[PositionSizeResult],
        context: PositionSizingContext,
        success: bool,
        rejection_reason: str,
        latency_ms: float,
    ) -> None:
        self.calls.append((result, context, success, rejection_reason, latency_ms))


@pytest.fixture
def base_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        policy_version="1.0.0",
        minimum_position_notional=10.0,
        maximum_position_notional=50000.0,
        minimum_quantity=0.001,
        maximum_quantity=100.0,
        maximum_leverage=10.0,
        maximum_margin_fraction=1.0,  # Relaxed to 100% of available balance
        maximum_symbol_exposure_fraction=1.0,  # Relaxed to 100% of equity
        maximum_portfolio_exposure_fraction=1.0,  # Relaxed to 100% of equity
        rounding_mode="DOWN",
        reject_if_below_min_quantity=True,
        reject_if_above_max_quantity=True,
        reject_if_stop_distance_invalid=True,
        reject_if_market_data_stale=True,
        market_data_max_age_seconds=10.0,
        authorization_max_age_seconds=10.0,
    )


@pytest.fixture
def base_proposal() -> TradeProposal:
    now_str = datetime.now(timezone.utc).isoformat()
    expires_str = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
    return TradeProposal(
        proposal_id="proposal-xyz-789",
        symbol="BTC/USDT",
        direction="BULLISH",
        confidence=0.80,
        fusion_score=0.80,
        source_model_version="gbdt-v1",
        fusion_policy_version="fusion-1.0",
        reasoning_request_id=None,
        created_at=now_str,
        expires_at=expires_str,
        risk_flags=[],
        metadata={},
    )


@pytest.fixture
def base_authorization() -> RiskAuthorizationResult:
    now_str = datetime.now(timezone.utc).isoformat()
    return RiskAuthorizationResult(
        authorization_id="auth-xyz-123",
        proposal_id="proposal-xyz-789",
        symbol="BTC/USDT",
        direction="BULLISH",
        status=RiskAuthorizationStatus.APPROVED,
        original_confidence=0.80,
        effective_confidence=0.80,
        rejection_reasons=[],
        adjustment_reasons=[],
        triggered_rules=[],
        policy_version="risk-1.0.0",
        source_model_version="gbdt-v1",
        fusion_policy_version="fusion-1.0",
        proposal_created_at=now_str,
        evaluated_at=now_str,
        latency_ms=1.5,
        requested_risk_fraction=0.01,
        authorized_risk_fraction=0.01,  # 1% equity risk
    )


@pytest.fixture
def base_context() -> PositionSizingContext:
    now_str = datetime.now(timezone.utc).isoformat()
    return PositionSizingContext(
        symbol="BTC/USDT",
        instrument_type="spot",
        equity=10000.0,
        available_balance=5000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,  # 1000 USDT stop distance (2% price drop)
        market_price=50000.0,
        leverage=1.0,
        contract_size=1.0,
        lot_size=0.001,
        min_quantity=0.001,
        max_quantity=10.0,
        quantity_step=0.001,
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=now_str,
        timestamp=now_str,
    )


# 1. Valid Bullish Sizing
def test_valid_bullish_sizing(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)
    result = engine.evaluate(base_proposal, base_authorization, base_context)

    # Calculation:
    # risk_amount = 10000 equity * 0.01 authorized_risk = 100 USDT
    # stop_distance_absolute = 50000 entry - 49000 stop = 1000 USDT
    # stop_distance_fraction = 1000 / 50000 = 0.02
    # position_notional = 100 / 0.02 = 5000 USDT
    # quantity_raw = 5000 / 50000 = 0.1 BTC
    # quantity_normalized = 0.1 (exact step match)
    # actual_notional = 5000.0
    # actual_risk = 100.0
    # estimated_margin = 5000.0 / 1.0 = 5000.0

    assert isinstance(result, PositionSizeResult)
    assert result.symbol == "BTC/USDT"
    assert result.direction == "BULLISH"
    assert result.quantity == 0.1
    assert result.position_notional == 5000.0
    assert result.entry_price == 50000.0
    assert result.stop_loss_price == 49000.0
    assert result.risk_amount == 100.0
    assert result.authorized_risk_fraction == 0.01
    assert result.leverage == 1.0
    assert result.estimated_margin_required == 5000.0
    assert result.policy_version == "1.0.0"
    assert result.authorization_id == "auth-xyz-123"
    assert result.proposal_id == "proposal-xyz-789"
    assert result.source_model_version == "gbdt-v1"


# 2. Valid Bearish Sizing
def test_valid_bearish_sizing(base_policy, base_proposal, base_authorization, base_context):
    # Set direction BEARISH
    proposal_bearish = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction="BEARISH",
        confidence=base_proposal.confidence,
        fusion_score=base_proposal.fusion_score,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=base_proposal.reasoning_request_id,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata=base_proposal.metadata,
    )
    auth_bearish = RiskAuthorizationResult(
        authorization_id=base_authorization.authorization_id,
        proposal_id=base_authorization.proposal_id,
        symbol=base_authorization.symbol,
        direction="BEARISH",
        status=base_authorization.status,
        original_confidence=base_authorization.original_confidence,
        effective_confidence=base_authorization.effective_confidence,
        rejection_reasons=base_authorization.rejection_reasons,
        adjustment_reasons=base_authorization.adjustment_reasons,
        triggered_rules=base_authorization.triggered_rules,
        policy_version=base_authorization.policy_version,
        source_model_version=base_authorization.source_model_version,
        fusion_policy_version=base_authorization.fusion_policy_version,
        proposal_created_at=base_authorization.proposal_created_at,
        evaluated_at=base_authorization.evaluated_at,
        latency_ms=base_authorization.latency_ms,
        requested_risk_fraction=base_authorization.requested_risk_fraction,
        authorized_risk_fraction=base_authorization.authorized_risk_fraction,
    )
    context_bearish = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=50000.0,
        stop_loss_price=51000.0,  # 1000 USDT stop distance above entry
        market_price=50000.0,
        leverage=5.0,  # leverage 5x
        contract_size=1.0,
        lot_size=0.001,
        min_quantity=0.001,
        max_quantity=10.0,
        quantity_step=0.001,
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )

    engine = PositionSizingEngine(policy=base_policy)
    result = engine.evaluate(proposal_bearish, auth_bearish, context_bearish)

    # risk_amount = 100 USDT
    # stop_distance_fraction = 1000 / 50000 = 0.02
    # position_notional = 100 / 0.02 = 5000 USDT
    # quantity = 0.1
    # margin = 5000 / 5.0 = 1000.0

    assert result.direction == "BEARISH"
    assert result.quantity == 0.1
    assert result.position_notional == 5000.0
    assert result.estimated_margin_required == 1000.0
    assert result.risk_amount == 100.0


# 3. Step Sizing Rounding DOWN Verification (Decimal precision)
def test_quantity_rounding_down(base_policy, base_proposal, base_authorization, base_context):
    # Set step to 0.01
    context_step = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        market_price=50000.0,
        leverage=1.0,
        contract_size=1.0,
        lot_size=0.01,
        min_quantity=0.01,
        max_quantity=10.0,
        quantity_step=0.01,  # 0.01 step
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )
    # Risk fraction 0.01053 -> risk_amount = 105.3 USDT
    # Stop distance fraction = 0.02
    # Notional raw = 105.3 / 0.02 = 5265.0 USDT
    # Qty raw = 5265.0 / 50000.0 = 0.1053 BTC
    # Rounded down to 0.01 step should be exactly 0.10 BTC (not 0.11, not 0.1053)
    auth = RiskAuthorizationResult(
        authorization_id="auth",
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        status=base_authorization.status,
        original_confidence=0.8,
        effective_confidence=0.8,
        rejection_reasons=[],
        adjustment_reasons=[],
        triggered_rules=[],
        policy_version="risk-1.0.0",
        source_model_version="gbdt-v1",
        fusion_policy_version="fusion-1.0",
        proposal_created_at=base_authorization.proposal_created_at,
        evaluated_at=base_authorization.evaluated_at,
        latency_ms=1.0,
        requested_risk_fraction=0.01053,
        authorized_risk_fraction=0.01053,
    )

    engine = PositionSizingEngine(policy=base_policy)
    result = engine.evaluate(base_proposal, auth, context_step)

    assert result.quantity == 0.1
    # Risk check: 0.1 qty * 50000 * 0.02 = 100 USDT actual risk <= 105.3 auth risk
    assert result.risk_amount == 100.0
    assert result.risk_amount <= 105.3


# 4. Rounding UP violates authorized risk limit
def test_rounding_up_risk_violation(base_policy, base_proposal, base_authorization, base_context):
    # If policy utilizes ROUND_UP, it might yield a risk exceeding authorized
    policy_up = PositionSizingPolicy(
        policy_version="policy-up",
        minimum_position_notional=10.0,
        maximum_position_notional=50000.0,
        minimum_quantity=0.001,
        maximum_quantity=100.0,
        maximum_leverage=10.0,
        maximum_margin_fraction=1.0,
        maximum_symbol_exposure_fraction=1.0,
        maximum_portfolio_exposure_fraction=1.0,
        rounding_mode="UP",  # UP
        reject_if_below_min_quantity=True,
        reject_if_above_max_quantity=True,
        reject_if_stop_distance_invalid=True,
        reject_if_market_data_stale=True,
        market_data_max_age_seconds=10.0,
        authorization_max_age_seconds=10.0,
    )
    # Let's check with a raw quantity of 0.1001 BTC
    # auth_risk = 0.01001 -> risk_amount = 100.1
    # notional = 100.1 / 0.02 = 5005.0 -> quantity = 0.1001
    # Rounding UP with 0.01 step -> 0.11 BTC -> actual risk = 110.0 USDT > 100.1 USDT
    auth = RiskAuthorizationResult(
        authorization_id="auth",
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        status=base_authorization.status,
        original_confidence=0.8,
        effective_confidence=0.8,
        rejection_reasons=[],
        adjustment_reasons=[],
        triggered_rules=[],
        policy_version="risk-1.0.0",
        source_model_version="gbdt-v1",
        fusion_policy_version="fusion-1.0",
        proposal_created_at=base_authorization.proposal_created_at,
        evaluated_at=base_authorization.evaluated_at,
        latency_ms=1.0,
        requested_risk_fraction=0.01001,
        authorized_risk_fraction=0.01001,
    )
    context_up = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        market_price=50000.0,
        leverage=1.0,
        contract_size=1.0,
        lot_size=0.01,
        min_quantity=0.01,
        max_quantity=10.0,
        quantity_step=0.01,
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )

    engine = PositionSizingEngine(policy=policy_up)
    # Must fail because rounded position risk (110.0) exceeds authorized risk (100.1)
    with pytest.raises(PositionLimitError) as exc_info:
        engine.evaluate(base_proposal, auth, context_up)
    assert "exceeds authorized limit" in str(exc_info.value)


# 5. Leverage does NOT increase risk budget
def test_leverage_does_not_increase_risk_budget(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # 1x leverage context
    context_1x = base_context
    result_1x = engine.evaluate(base_proposal, base_authorization, context_1x)

    # 10x leverage context
    context_10x = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=10.0,  # 10x leverage
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=base_context.current_symbol_exposure,
        current_portfolio_exposure=base_context.current_portfolio_exposure,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )
    result_10x = engine.evaluate(base_proposal, base_authorization, context_10x)

    # Both results must have the EXACT SAME risk_amount (100 USDT)
    # The only difference is the estimated margin: 1x margin = 5000, 10x margin = 500
    assert result_1x.risk_amount == 100.0
    assert result_10x.risk_amount == 100.0
    assert result_1x.quantity == 0.1
    assert result_10x.quantity == 0.1
    assert result_1x.estimated_margin_required == 5000.0
    assert result_10x.estimated_margin_required == 500.0


# 6. Stop-Loss Direction Violations
def test_invalid_stop_orientation(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # Bullish proposal, but stop-loss >= entry
    bad_context_bull = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=50000.0,
        stop_loss_price=50100.0,  # invalid for bullish
        market_price=50000.0,
        leverage=1.0,
        contract_size=1.0,
        lot_size=0.001,
        min_quantity=0.001,
        max_quantity=10.0,
        quantity_step=0.001,
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )
    with pytest.raises(InvalidStopDistanceError):
        engine.evaluate(base_proposal, base_authorization, bad_context_bull)

    # Zero stop distance
    zero_context = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=50000.0,
        stop_loss_price=50000.0,  # zero stop distance
        market_price=50000.0,
        leverage=1.0,
        contract_size=1.0,
        lot_size=0.001,
        min_quantity=0.001,
        max_quantity=10.0,
        quantity_step=0.001,
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )
    with pytest.raises(InvalidStopDistanceError):
        engine.evaluate(base_proposal, base_authorization, zero_context)


# 7. Sizing limits (Min/Max quantity, Min/Max notional)
def test_sizing_limits_gates(base_policy, base_proposal, base_authorization, base_context):
    # Minimum quantity gate check (raw quantity = 0.1, min policy is 0.5)
    policy_min_qty = PositionSizingPolicy(
        policy_version="1.0.0",
        minimum_position_notional=10.0,
        maximum_position_notional=50000.0,
        minimum_quantity=0.5,  # higher min quantity
        maximum_quantity=100.0,
        maximum_leverage=10.0,
        maximum_margin_fraction=0.50,
        maximum_symbol_exposure_fraction=0.20,
        maximum_portfolio_exposure_fraction=0.50,
        rounding_mode="DOWN",
        reject_if_below_min_quantity=True,
        reject_if_above_max_quantity=True,
        reject_if_stop_distance_invalid=True,
        reject_if_market_data_stale=True,
        market_data_max_age_seconds=10.0,
        authorization_max_age_seconds=10.0,
    )
    engine = PositionSizingEngine(policy=policy_min_qty)
    with pytest.raises(PositionLimitError):
        engine.evaluate(base_proposal, base_authorization, base_context)


# 8. Exposure cap limit checks (Symbol & Portfolio)
def test_exposure_limit_violations(base_policy, base_proposal, base_authorization, base_context):
    policy_restricted = PositionSizingPolicy(
        policy_version="1.0.0",
        minimum_position_notional=10.0,
        maximum_position_notional=50000.0,
        minimum_quantity=0.001,
        maximum_quantity=100.0,
        maximum_leverage=10.0,
        maximum_margin_fraction=1.0,
        maximum_symbol_exposure_fraction=0.20,  # 20% max sym cap
        maximum_portfolio_exposure_fraction=0.50,
        rounding_mode="DOWN",
        reject_if_below_min_quantity=True,
        reject_if_above_max_quantity=True,
        reject_if_stop_distance_invalid=True,
        reject_if_market_data_stale=True,
        market_data_max_age_seconds=10.0,
        authorization_max_age_seconds=10.0,
    )
    engine = PositionSizingEngine(policy=policy_restricted)

    # Symbol exposure fraction limit is 20% -> 2000 USDT on 10000 equity.
    # Current symbol exposure is 1000. Evaluated position is 5000.
    # Total symbol exposure = 6000 > 2000 -> REJECT
    context_exposure = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=base_context.leverage,
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=1000.0,       # 1000 already exposed
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )

    with pytest.raises(PositionLimitError) as exc_info:
        engine.evaluate(base_proposal, base_authorization, context_exposure)
    assert "Symbol exposure" in str(exc_info.value)


# 9. Available margin deficit check
def test_insufficient_margin_capital(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # margin required is 5000 (1x leverage). Available balance is only 1000.
    context_no_cap = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=1000.0,  # low capital
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=base_context.leverage,
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )
    with pytest.raises(InsufficientCapitalError):
        engine.evaluate(base_proposal, base_authorization, context_no_cap)


# 10. Stale market data
def test_stale_market_data(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # Market timestamp is 15 seconds older than current timestamp (limit is 10s)
    current_time = datetime.now(timezone.utc)
    market_time = current_time - timedelta(seconds=15)
    
    context_stale = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=base_context.leverage,
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=market_time.isoformat(),
        timestamp=current_time.isoformat(),
    )
    with pytest.raises(PositionSizingValidationError) as exc_info:
        engine.evaluate(base_proposal, base_authorization, context_stale)
    assert "Stale market data" in str(exc_info.value)


# 11. Stale / Expired Authorization Result
def test_stale_authorization(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # Authorization was evaluated 20s ago relative to context execution timestamp (limit 10s)
    context_time = datetime.now(timezone.utc)
    auth_evaluated_time = context_time - timedelta(seconds=20)

    auth = RiskAuthorizationResult(
        authorization_id=base_authorization.authorization_id,
        proposal_id=base_authorization.proposal_id,
        symbol=base_authorization.symbol,
        direction=base_authorization.direction,
        status=base_authorization.status,
        original_confidence=base_authorization.original_confidence,
        effective_confidence=base_authorization.effective_confidence,
        rejection_reasons=base_authorization.rejection_reasons,
        adjustment_reasons=base_authorization.adjustment_reasons,
        triggered_rules=base_authorization.triggered_rules,
        policy_version=base_authorization.policy_version,
        source_model_version=base_authorization.source_model_version,
        fusion_policy_version=base_authorization.fusion_policy_version,
        proposal_created_at=base_authorization.proposal_created_at,
        evaluated_at=auth_evaluated_time.isoformat(),
        latency_ms=base_authorization.latency_ms,
        requested_risk_fraction=base_authorization.requested_risk_fraction,
        authorized_risk_fraction=base_authorization.authorized_risk_fraction,
    )
    
    context = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=base_context.leverage,
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=context_time.isoformat(),
        timestamp=context_time.isoformat(),
    )

    with pytest.raises(AuthorizationError) as exc_info:
        engine.evaluate(base_proposal, auth, context)
    assert "Authorization is stale" in str(exc_info.value)


# 12. Rejected status is rejected
def test_rejected_auth_raises_error(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    auth = RiskAuthorizationResult(
        authorization_id=base_authorization.authorization_id,
        proposal_id=base_authorization.proposal_id,
        symbol=base_authorization.symbol,
        direction=base_authorization.direction,
        status=RiskAuthorizationStatus.REJECTED,  # REJECTED status
        original_confidence=base_authorization.original_confidence,
        effective_confidence=base_authorization.effective_confidence,
        rejection_reasons=["High volatility"],
        adjustment_reasons=[],
        triggered_rules=["VOLATILITY_GATE"],
        policy_version=base_authorization.policy_version,
        source_model_version=base_authorization.source_model_version,
        fusion_policy_version=base_authorization.fusion_policy_version,
        proposal_created_at=base_authorization.proposal_created_at,
        evaluated_at=base_authorization.evaluated_at,
        latency_ms=1.0,
        requested_risk_fraction=0.01,
        authorized_risk_fraction=0.0,
    )
    with pytest.raises(AuthorizationError):
        engine.evaluate(base_proposal, auth, base_context)


# 13. Adjusted status sizing uses final auth fraction
def test_adjusted_auth_uses_reduced_fraction(base_policy, base_proposal, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # Adjusted risk fraction has been reduced to 0.005 by RiskGuardEngine
    auth_adjusted = RiskAuthorizationResult(
        authorization_id="auth",
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        status=RiskAuthorizationStatus.ADJUSTED,
        original_confidence=0.8,
        effective_confidence=0.8,
        rejection_reasons=[],
        adjustment_reasons=["Reduced via Volatility rules"],
        triggered_rules=["VOLATILITY_ADJUSTMENT"],
        policy_version="risk-1.0.0",
        source_model_version="gbdt-v1",
        fusion_policy_version="fusion-1.0",
        proposal_created_at=datetime.now(timezone.utc).isoformat(),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        latency_ms=1.0,
        requested_risk_fraction=0.01,
        authorized_risk_fraction=0.005,  # Reduced risk fraction
    )

    result = engine.evaluate(base_proposal, auth_adjusted, base_context)
    # risk_amount must be 50.0 USDT (0.005 of 10000 equity)
    assert result.authorized_risk_fraction == 0.005
    assert result.risk_amount == 50.0
    assert result.quantity == 0.05


# 14. NEUTRAL direction check (fail-closed)
def test_neutral_direction_fail_closed(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)
    
    proposal_neutral = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction="NEUTRAL",
        confidence=base_proposal.confidence,
        fusion_score=base_proposal.fusion_score,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=base_proposal.reasoning_request_id,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata=base_proposal.metadata,
    )
    with pytest.raises(PositionSizingValidationError):
        engine.evaluate(proposal_neutral, base_authorization, base_context)


# 15. Future market timestamp rejection
def test_future_market_timestamp(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    # Market timestamp is 5 seconds in the FUTURE compared to current timestamp
    current_time = datetime.now(timezone.utc)
    market_time = current_time + timedelta(seconds=5)

    context_future = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type=base_context.instrument_type,
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=base_context.leverage,
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=market_time.isoformat(),
        timestamp=current_time.isoformat(),
    )
    with pytest.raises(PositionSizingValidationError):
        engine.evaluate(base_proposal, base_authorization, context_future)


# 16. NaN / Inf rejection
def test_nan_inf_inputs_rejected(base_policy, base_proposal, base_authorization):
    # Try constructing context with NaN
    with pytest.raises(PositionSizingValidationError):
        PositionSizingContext(
            symbol="BTC/USDT",
            instrument_type="spot",
            equity=float("nan"),  # NaN
            available_balance=5000.0,
            entry_price=50000.0,
            stop_loss_price=49000.0,
            market_price=50000.0,
            leverage=1.0,
            contract_size=1.0,
            lot_size=0.001,
            min_quantity=0.001,
            max_quantity=10.0,
            quantity_step=0.001,
            price_tick=0.01,
            current_symbol_exposure=0.0,
            current_portfolio_exposure=0.0,
            market_timestamp=datetime.now(timezone.utc).isoformat(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# 17. Telemetry execution
def test_telemetry_captures_audit_info(base_policy, base_proposal, base_authorization, base_context):
    mock_telemetry = MockPositionSizingTelemetrySink()
    engine = PositionSizingEngine(policy=base_policy, telemetry_sink=mock_telemetry)

    result = engine.evaluate(base_proposal, base_authorization, base_context)
    
    assert len(mock_telemetry.calls) == 1
    logged_result, logged_context, success, reason, latency = mock_telemetry.calls[0]
    
    assert success is True
    assert logged_result == result
    assert logged_context == base_context
    assert reason == ""
    assert latency > 0.0


# 18. Deterministic Repeated Evaluations
def test_sizing_determinism(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)
    
    result_1 = engine.evaluate(base_proposal, base_authorization, base_context)
    result_2 = engine.evaluate(base_proposal, base_authorization, base_context)

    # UUID and timestamp will naturally differ, but the economics must match perfectly
    assert result_1.quantity == result_2.quantity
    assert result_1.position_notional == result_2.position_notional
    assert result_1.risk_amount == result_2.risk_amount
    assert result_1.estimated_margin_required == result_2.estimated_margin_required


# 19. Unsupported Instrument type behavior
def test_unsupported_instrument_fails_closed(base_policy, base_proposal, base_authorization, base_context):
    engine = PositionSizingEngine(policy=base_policy)

    context_unsupported = PositionSizingContext(
        symbol=base_context.symbol,
        instrument_type="option",  # unsupported type
        equity=base_context.equity,
        available_balance=base_context.available_balance,
        entry_price=base_context.entry_price,
        stop_loss_price=base_context.stop_loss_price,
        market_price=base_context.market_price,
        leverage=base_context.leverage,
        contract_size=base_context.contract_size,
        lot_size=base_context.lot_size,
        min_quantity=base_context.min_quantity,
        max_quantity=base_context.max_quantity,
        quantity_step=base_context.quantity_step,
        price_tick=base_context.price_tick,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=base_context.market_timestamp,
        timestamp=base_context.timestamp,
    )
    with pytest.raises(PositionSizingValidationError):
        engine.evaluate(base_proposal, base_authorization, context_unsupported)


# 20. Execution Import Isolation Verification
def test_positioning_package_execution_isolation():
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
        "ReasoningEngine",
        "OllamaProvider",
        "ollama",
    ]

    positioning_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "backend",
        "positioning"
    )

    python_files = [
        os.path.join(positioning_dir, f)
        for f in os.listdir(positioning_dir)
        if f.endswith(".py")
    ]

    for filepath in python_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # AST analysis
        tree = ast.parse(content, filename=filepath)
        for node in ast.walk(tree):
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
                    for alias in node.names:
                        assert keyword not in alias.name, (
                            f"Forbidden import name '{alias.name}' from '{node.module}' in {filepath}."
                        )
