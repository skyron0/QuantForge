import os
import ast
import math
import uuid
import pytest
import threading
import time
import dataclasses
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from backend.decision.models import TradeProposal
from backend.risk.models import RiskAuthorizationResult, RiskAuthorizationStatus
from backend.positioning.models import PositionSizeResult

from backend.execution_authorization import (
    ExecutionEnvironment,
    OrderDirection,
    OrderType,
    ExecutionAuthorizationStatus,
    ExecutionContext,
    OrderIntent,
    ExecutionAuthorizationResult as EngineResult,
    ExecutionPolicy,
    IdempotencyStore,
    ConsoleExecutionAuthorizationTelemetrySink,
    ExecutionAuthorizationEngine,
    ExecutionAuthorizationError,
    InvalidExecutionPolicyError,
    ExecutionValidationError,
    LineageMismatchError,
    ExecutionDisabledError,
    KillSwitchActiveError,
    SymbolTradingDisabledError,
    StaleMarketDataError,
    DuplicateIntentError,
    InvalidOrderIntentError,
    LiveExecutionNotAllowedError,
)


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def base_proposal() -> TradeProposal:
    return TradeProposal(
        proposal_id="proposal-123",
        symbol="BTC/USDT",
        direction="BULLISH",
        confidence=0.85,
        fusion_score=0.90,
        source_model_version="ml-v2",
        fusion_policy_version="fusion-v1",
        reasoning_request_id="req-999",
        created_at="2026-07-21T12:00:00Z",
        expires_at="2026-07-21T12:05:00Z",
        risk_flags=[],
        metadata={"order_type": "MARKET"}
    )


@pytest.fixture
def base_risk_auth() -> RiskAuthorizationResult:
    return RiskAuthorizationResult(
        authorization_id="auth-risk-456",
        proposal_id="proposal-123",
        symbol="BTC/USDT",
        direction="BULLISH",
        status=RiskAuthorizationStatus.APPROVED,
        original_confidence=0.85,
        effective_confidence=0.85,
        rejection_reasons=[],
        adjustment_reasons=[],
        triggered_rules=[],
        policy_version="risk-policy-v1",
        source_model_version="ml-v2",
        fusion_policy_version="fusion-v1",
        proposal_created_at="2026-07-21T12:00:00Z",
        evaluated_at="2026-07-21T12:00:01Z",
        latency_ms=2.0,
        requested_risk_fraction=0.01,
        authorized_risk_fraction=0.01,
    )


@pytest.fixture
def base_size_res() -> PositionSizeResult:
    return PositionSizeResult(
        sizing_id="sizing-789",
        authorization_id="auth-risk-456",
        proposal_id="proposal-123",
        symbol="BTC/USDT",
        direction="BULLISH",
        quantity=0.1,
        position_notional=5000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,  # 2% stop distance
        stop_distance_absolute=1000.0,
        stop_distance_fraction=0.02,
        risk_amount=100.0,
        authorized_risk_fraction=0.01,
        leverage=1.0,
        estimated_margin_required=5000.0,
        policy_version="sizing-policy-v1",
        created_at="2026-07-21T12:00:02Z",
        source_model_version="ml-v2",
        metadata={}
    )


@pytest.fixture
def base_context() -> ExecutionContext:
    return ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp="2026-07-21T12:00:03Z",
        market_timestamp="2026-07-21T12:00:03Z",
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0,
        metadata={}
    )


@pytest.fixture
def base_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        policy_version="exec-policy-1.0",
        allowed_environments=[ExecutionEnvironment.PAPER, ExecutionEnvironment.SHADOW],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET, OrderType.LIMIT],
        allow_live_execution_intents=False,  # default safety
        require_execution_enabled=True,
        reject_when_kill_switch_active=True,
        require_symbol_enabled=True,
        maximum_clock_skew_seconds=5.0
    )


@pytest.fixture
def store() -> IdempotencyStore:
    return IdempotencyStore(max_keys=100)


@pytest.fixture
def engine(base_policy, store) -> ExecutionAuthorizationEngine:
    sink = ConsoleExecutionAuthorizationTelemetrySink()
    return ExecutionAuthorizationEngine(
        policy=base_policy,
        idempotency_store=store,
        telemetry_sink=sink
    )


# ==========================================
# Unit Tests
# ==========================================

# 1. Environment Gating
def test_valid_paper_authorization(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, base_context)
    assert res.status == ExecutionAuthorizationStatus.AUTHORIZED
    assert res.intent is not None
    assert res.intent.environment == ExecutionEnvironment.PAPER
    assert res.intent.quantity == 0.1
    assert res.intent.direction == OrderDirection.BUY
    assert res.latency_ms > 0


def test_valid_shadow_authorization(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    ctx_shadow = ExecutionContext(
        environment=ExecutionEnvironment.SHADOW,
        current_timestamp=base_context.current_timestamp,
        market_timestamp=base_context.market_timestamp,
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, ctx_shadow)
    assert res.status == ExecutionAuthorizationStatus.AUTHORIZED
    assert res.intent is not None
    assert res.intent.environment == ExecutionEnvironment.SHADOW


def test_live_rejected_by_default(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # Even if environment is added to allowed environments, allow_live_execution_intents is False by default
    policy_with_live = ExecutionPolicy(
        policy_version="exec-policy-live",
        allowed_environments=[ExecutionEnvironment.LIVE],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET],
        allow_live_execution_intents=False,  # FAIL CLOSED DEFAULT
    )
    eng_live = ExecutionAuthorizationEngine(policy=policy_with_live, idempotency_store=IdempotencyStore())
    ctx_live = ExecutionContext(
        environment=ExecutionEnvironment.LIVE,
        current_timestamp=base_context.current_timestamp,
        market_timestamp=base_context.market_timestamp,
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = eng_live.evaluate(base_proposal, base_risk_auth, base_size_res, ctx_live)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Live execution intent is rejected" in res.rejection_reason
    assert "LIVE_NOT_PERMITTED" in res.triggered_rules or "LiveExecutionNotAllowedError" in res.triggered_rules


def test_explicitly_enabled_live_authorization(base_proposal, base_risk_auth, base_size_res, base_context):
    policy_live = ExecutionPolicy(
        policy_version="exec-policy-live-ok",
        allowed_environments=[ExecutionEnvironment.LIVE],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET],
        allow_live_execution_intents=True,  # Double opt-in 1
    )
    eng_live = ExecutionAuthorizationEngine(policy=policy_live, idempotency_store=IdempotencyStore())
    ctx_live = ExecutionContext(
        environment=ExecutionEnvironment.LIVE,  # Double opt-in 2
        current_timestamp=base_context.current_timestamp,
        market_timestamp=base_context.market_timestamp,
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = eng_live.evaluate(base_proposal, base_risk_auth, base_size_res, ctx_live)
    assert res.status == ExecutionAuthorizationStatus.AUTHORIZED
    assert res.intent is not None
    assert res.intent.environment == ExecutionEnvironment.LIVE


# 2. System and Status Gates
def test_execution_disabled_rejection(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    ctx_disabled = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp=base_context.current_timestamp,
        market_timestamp=base_context.market_timestamp,
        execution_enabled=False,  # DISABLED
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, ctx_disabled)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Global execution is disabled" in res.rejection_reason
    assert "EXECUTION_DISABLED" in res.triggered_rules


def test_kill_switch_rejection(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    ctx_kill = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp=base_context.current_timestamp,
        market_timestamp=base_context.market_timestamp,
        execution_enabled=True,
        kill_switch_active=True,  # KILL SWITCH
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, ctx_kill)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Kill switch is active" in res.rejection_reason
    assert "KILL_SWITCH" in res.triggered_rules


def test_disabled_symbol_rejection(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    ctx_sym_disabled = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp=base_context.current_timestamp,
        market_timestamp=base_context.market_timestamp,
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=False,  # SYMBOL DISABLED
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, ctx_sym_disabled)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Symbol trading is disabled" in res.rejection_reason
    assert "SYMBOL_DISABLED" in res.triggered_rules


# 3. Lineage and Consistency Validations
def test_lineage_proposal_mismatch(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # Modify proposal_id in risk_auth
    mismatch_risk = dataclasses.replace(base_risk_auth, proposal_id="proposal-mismatch-xxx")
    res = engine.evaluate(base_proposal, mismatch_risk, base_size_res, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "proposal ID" in res.rejection_reason
    assert "LineageMismatchError" in res.triggered_rules


def test_lineage_risk_auth_id_mismatch(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # Modify authorization_id reference in sizing result
    mismatch_sizing = dataclasses.replace(base_size_res, authorization_id="auth-mismatch-xxx")
    res = engine.evaluate(base_proposal, base_risk_auth, mismatch_sizing, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "auth ID" in res.rejection_reason
    assert "LineageMismatchError" in res.triggered_rules


def test_lineage_symbol_mismatch(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # Proposal symbol mismatch
    bad_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol="ETH/USDT",  # Mismatch symbol BTC vs ETH
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=base_proposal.fusion_score,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=base_proposal.reasoning_request_id,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata=base_proposal.metadata
    )
    res = engine.evaluate(bad_proposal, base_risk_auth, base_size_res, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Symbol mismatch" in res.rejection_reason
    assert "LineageMismatchError" in res.triggered_rules


def test_lineage_direction_mismatch(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # Adjust sizing result direction to BEARISH
    mismatch_direction_sizing = dataclasses.replace(base_size_res, direction="BEARISH")
    res = engine.evaluate(base_proposal, base_risk_auth, mismatch_direction_sizing, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Direction mismatch" in res.rejection_reason
    assert "LineageMismatchError" in res.triggered_rules


# 4. Input Validations & Quantity Checks
def test_zero_or_negative_quantity(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    bad_sizing = dataclasses.replace(base_size_res, quantity=0.0)
    res = engine.evaluate(base_proposal, base_risk_auth, bad_sizing, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Sizing quantity" in res.rejection_reason


def test_nan_quantity_validation():
    with pytest.raises(ExecutionValidationError):
        OrderIntent(
            intent_id="intent-1",
            idempotency_key="key-1",
            proposal_id="p-1",
            risk_authorization_id="r-1",
            sizing_id="s-1",
            symbol="BTC/USDT",
            direction=OrderDirection.BUY,
            quantity=float("nan"),  # NaN
            order_type=OrderType.MARKET,
            limit_price=None,
            stop_loss=None,
            take_profit=None,
            environment=ExecutionEnvironment.PAPER,
            source_model_version="1",
            fusion_policy_version="1",
            risk_policy_version="1",
            position_sizing_policy_version="1",
            execution_policy_version="1",
            reasoning_request_id=None,
            created_at="2026-07-21T12:00:00Z",
            expires_at="2026-07-21T12:05:00Z"
        )


def test_actual_risk_exceeding_authorized_risk(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # Sizing actual risk is 0.05, but Risk authorization is only 0.01
    bad_sizing = dataclasses.replace(base_size_res, authorized_risk_fraction=0.05, risk_amount=500.0, quantity=0.5)
    res = engine.evaluate(base_proposal, base_risk_auth, bad_sizing, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "exceeds authorized RiskGuard fraction" in res.rejection_reason


# 5. Direction and Mapping Validations
def test_bullish_bearish_neutral_mapping(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # BEARISH -> SELL
    bearish_proposal = dataclasses.replace(base_proposal, direction="BEARISH")
    bearish_risk = dataclasses.replace(base_risk_auth, direction="BEARISH")
    bearish_sizing = dataclasses.replace(base_size_res, direction="BEARISH", stop_loss_price=51000.0)
    res = engine.evaluate(bearish_proposal, bearish_risk, bearish_sizing, base_context)
    assert res.status == ExecutionAuthorizationStatus.AUTHORIZED
    assert res.intent is not None
    assert res.intent.direction == OrderDirection.SELL

    # NEUTRAL -> Rejected
    neutral_proposal = dataclasses.replace(base_proposal, direction="NEUTRAL")
    neutral_risk = dataclasses.replace(base_risk_auth, direction="NEUTRAL")
    neutral_sizing = dataclasses.replace(base_size_res, direction="NEUTRAL")
    res_neut = engine.evaluate(neutral_proposal, neutral_risk, neutral_sizing, base_context)
    assert res_neut.status == ExecutionAuthorizationStatus.REJECTED
    assert "NEUTRAL proposals are rejected" in res_neut.rejection_reason


# 6. Order Type Constraints
def test_market_order_with_limit_price_rejection(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    proposal_market_with_price = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=base_proposal.fusion_score,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=base_proposal.reasoning_request_id,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata={"order_type": "MARKET", "limit_price": 50100.0}  # LIMIT price in MARKET order
    )
    res = engine.evaluate(proposal_market_with_price, base_risk_auth, base_size_res, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "MARKET order cannot contain a limit price" in res.rejection_reason


def test_limit_order_without_limit_price_rejection(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    proposal_limit_no_price = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=base_proposal.fusion_score,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=base_proposal.reasoning_request_id,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata={"order_type": "LIMIT"}  # Mising limit price
    )
    res = engine.evaluate(proposal_limit_no_price, base_risk_auth, base_size_res, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "LIMIT order requires a limit price" in res.rejection_reason


# 7. Stop-Loss and Take-Profit Directional Orientations
def test_stop_loss_orientation_violations(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # BUY stop loss must be < entry price. Here we set it to 50500.0 (entry = 50000.0)
    bad_stop_sizing = dataclasses.replace(base_size_res, stop_loss_price=50500.0)
    res = engine.evaluate(base_proposal, base_risk_auth, bad_stop_sizing, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Stop loss" in res.rejection_reason


def test_take_profit_orientation_violations(engine, base_proposal, base_risk_auth, base_size_res, base_context):
    # BUY take profit must be > entry price (50000.0). We set it to 49500.0.
    policy_with_tp = ExecutionPolicy(
        policy_version="exec-policy-tp",
        allowed_environments=[ExecutionEnvironment.PAPER],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=False,
        require_take_profit=True,  # REQUIRED TP
        allowed_order_types=[OrderType.MARKET]
    )
    eng_tp = ExecutionAuthorizationEngine(policy=policy_with_tp, idempotency_store=IdempotencyStore())
    bad_tp_proposal = TradeProposal(
        proposal_id=base_proposal.proposal_id,
        symbol=base_proposal.symbol,
        direction=base_proposal.direction,
        confidence=base_proposal.confidence,
        fusion_score=base_proposal.fusion_score,
        source_model_version=base_proposal.source_model_version,
        fusion_policy_version=base_proposal.fusion_policy_version,
        reasoning_request_id=base_proposal.reasoning_request_id,
        created_at=base_proposal.created_at,
        expires_at=base_proposal.expires_at,
        risk_flags=base_proposal.risk_flags,
        metadata={"order_type": "MARKET", "take_profit": 49500.0}  # BUY TP below entry
    )
    res = eng_tp.evaluate(bad_tp_proposal, base_risk_auth, base_size_res, base_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Take profit" in res.rejection_reason


# 8. Freshness Gates
def test_stale_market_data_gate(engine, base_proposal, base_risk_auth, base_size_res):
    # Market timestamp is 2026-07-21T12:00:00Z, context time is 12:00:20Z (20s stale > max 10.0s)
    stale_context = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp="2026-07-21T12:00:20Z",
        market_timestamp="2026-07-21T12:00:00Z",
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, stale_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "Stale market data" in res.rejection_reason
    assert "STALE_MARKET_DATA" in res.triggered_rules


def test_future_market_data_skew_gate(engine, base_proposal, base_risk_auth, base_size_res):
    # Market data is 12:00:10Z, current time is 12:00:00Z (10s future skew > max 5.0s)
    future_context = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp="2026-07-21T12:00:00Z",
        market_timestamp="2026-07-21T12:00:10Z",
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, future_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "future" in res.rejection_reason
    assert "CLOCK_SKEW_LIMIT" in res.triggered_rules


def test_expired_proposal_rejection(engine, base_proposal, base_risk_auth, base_size_res):
    # Current timestamp is 12:06:00Z, proposal expired at 12:05:00Z
    expired_context = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp="2026-07-21T12:06:00Z",
        market_timestamp="2026-07-21T12:06:00Z",
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=50000.0
    )
    res = engine.evaluate(base_proposal, base_risk_auth, base_size_res, expired_context)
    assert res.status == ExecutionAuthorizationStatus.REJECTED
    assert "expired" in res.rejection_reason
    assert "PROPOSAL_EXPIRED" in res.triggered_rules


# 9. Idempotency Store Logic & Rollback
def test_idempotency_store_bounds_ttl_rollback(store):
    store.clear()
    assert len(store) == 0

    # 1. Register absent
    res1 = store.register_if_absent("key-1", {"val": 1}, ttl_seconds=1.0)
    assert res1 is True
    assert len(store) == 1

    # 2. Register repeat
    res2 = store.register_if_absent("key-1", {"val": 2}, ttl_seconds=1.0)
    assert res2 is False

    # 3. Retrieve
    assert store.get("key-1") == {"val": 1}

    # 4. Invalidate (rollback)
    store.invalidate("key-1")
    assert store.get("key-1") is None
    assert len(store) == 0

    # 5. capacity limits
    store_small = IdempotencyStore(max_keys=2)
    assert store_small.register_if_absent("k-1", 1, 10.0) is True
    assert store_small.register_if_absent("k-2", 2, 10.0) is True
    assert store_small.register_if_absent("k-3", 3, 10.0) is False  # Bounded check

    # 6. TTL expiry
    store_ttl = IdempotencyStore()
    assert store_ttl.register_if_absent("k-temp", 100, ttl_seconds=0.1) is True
    time.sleep(0.15)
    # Registering again after TTL expiry should work
    assert store_ttl.register_if_absent("k-temp", 200, ttl_seconds=10.0) is True
    assert store_ttl.get("k-temp") == 200


def test_idempotency_store_rollback_on_construction_failure(store, base_proposal, base_risk_auth, base_size_res, base_context):
    store.clear()
    policy = ExecutionPolicy(
        policy_version="exec-policy-rollback",
        allowed_environments=[ExecutionEnvironment.PAPER],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.01,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET]
    )

    # Let's break context formatting to cause an expected construction error inside evaluate()'s try-block during OrderIntent instantiation
    # For example, setting proposal_id to a value that isn't a string? No, OrderIntent doesn't restrict proposal_id to not be None, but direction check will break
    # Let's mock a context with current_timestamp = "invalid-date" so that datetime.fromisoformat raises ValueError inside OrderIntent block:
    # Actually, we can generate a situation where OrderIntent constructor fails, e.g. direction = "UNKNOWN"
    # Wait: if we pass mapped_direction = OrderDirection("xyz"), it raises ValueError.
    # In authorization.py:
    # intent = OrderIntent(..., direction=mapped_direction, quantity=size_res.quantity)
    # But wait, mapped_direction is validated first.
    # Let's pass a value that triggers validation inside OrderIntent constructor: e.g. quantity = -5.0.
    # But quantity <= 0 is checked at sizing validation:
    # `if size_res.quantity <= 0: raise ExecutionValidationError(...)`
    # How can we bypass sizing validation but fail inside OrderIntent?
    # Ah, let's look at `OrderIntent` constructor validations:
    # `if not isinstance(self.direction, OrderDirection): raise ExecutionValidationError(...)`
    # Yes, we could mock the mapped_direction choice. But wait: a simpler way is to trigger an exception during expiration date parse, or similar!
    # Yes, let's verify if `created_at` timestamp is fine in sizing checks, but contains an invalid value that fails inside the OrderIntent construct block.
    # Let's look at authorization.py line 287-294:
    # `created_dt = parse_iso(created_at)`
    # So if `context.current_timestamp` is valid in the first check, but gets modified? No, context is immutable.
    # What if `size_res.quantity` is valid, but one of the policy variables (e.g. `policy.order_intent_ttl_seconds` is NaN or causes overflow)?
    # Wait, we can test rollback by mocking or by passing a value that fails during OrderIntent construction, such as setting a non-numeric or float NaN value to `proposal.source_model_version`. Wait, `proposal.source_model_version` is a string, if we pass `None`? Yes, passing `None` might raise a validation or type error if we check it.
    # Let's check `OrderIntent.__post_init__`:
    # `for num_name in ["quantity", "limit_price", "stop_loss", "take_profit"]:`
    # `val = getattr(self, num_name)`
    # `if val is not None:`
    # `  if not isinstance(val, (int, float)): raise ExecutionValidationError(...)`
    # What if `proposal.metadata.get("take_profit", None)` returns `"string"`?
    # Sizing validation doesn't check `take_profit` type!
    # Yes! Sizing validation check doesn't check if `proposal.metadata.get("take_profit")` is a string! It checks `if bp_tp is not None:` and then does orientation checks or numeric checks.
    # Wait, in `authorization.py` we have:
    # `if math.isnan(bp_tp) or math.isinf(bp_tp) or bp_tp <= 0:`
    # This will check `math.isnan(bp_tp)` which raises TypeError if `bp_tp` is a string!
    # So if `take_profit` is a string (e.g. `"not-a-number"`), `math.isnan()` raises TypeError!
    # Since `TypeError` is raised inside the try-block, it will be caught, but wait: the reservation is done at line 273, with final OrderIntent created at 296!
    # Ah! If `TypeError` is raised *before* the reservation, the reservation won't even happen. We want the failure to happen *inside* the OrderIntent block (lines 285-319), which is protected by a nested try-except catching `construction_error` and triggering rollback on line 322!
    # Yes! So the exception must occur inside the `OrderIntent` constructor or inside lines 286-318.
    # Where? Line 289: `created_dt = parse_iso(created_at)`.
    # Wait! If `created_dt` parses successfully, what about `expires_dt`?
    # What if `self.policy.order_intent_ttl_seconds` is extremely large causing OverflowError or similar? That's hard to trigger cleanly.
    # What if `OrderIntent` metadata causes error?
    # What if we pass a proposal whose `metadata` is a mock object that raises an error when accessed, or `proposal.metadata = None`? Wait, `metadata` defaults to Dict, if `proposal.metadata` is `None`, then `proposal.metadata or {}` will evaluate to `{}`.
    # What if `proposal.reasoning_request_id` causes an error?
    # What if we mock `OrderIntent` class itself? Python allows us to mock the constructor of `OrderIntent` using `unittest.mock.patch`!
    # Yes! We can patch `backend.execution_authorization.authorization.OrderIntent` to raise a `RuntimeError("Simulated construction error")`!
    # This is elegant, standard, and doesn't rely on abusing properties.
    from unittest.mock import patch

    eng_mock = ExecutionAuthorizationEngine(policy=policy, idempotency_store=store)

    idempotency_key = f"{base_proposal.proposal_id}:{base_risk_auth.authorization_id}:{base_size_res.sizing_id}:{base_context.environment.value}"

    with patch("backend.execution_authorization.authorization.OrderIntent", side_effect=ValueError("Simulated intent construction error")):
        res = eng_mock.evaluate(base_proposal, base_risk_auth, base_size_res, base_context)
        assert res.status == ExecutionAuthorizationStatus.REJECTED
        assert "Simulated intent construction error" in res.rejection_reason

    # The idempotency key should NOT be reserved in the store because construction failed and triggered a rollback!
    assert store.get(idempotency_key) is None


# 10. Concurrency Thread Safety Tests
def test_simultaneous_authorization_attempts_concurrency(store, base_proposal, base_risk_auth, base_size_res, base_context, base_policy):
    store.clear()
    eng = ExecutionAuthorizationEngine(policy=base_policy, idempotency_store=store)

    results: List[EngineResult] = []
    threads = []
    lock = threading.Lock()

    def worker():
        res = eng.evaluate(base_proposal, base_risk_auth, base_size_res, base_context)
        with lock:
            results.append(res)

    # Launch 20 concurrent threads attempting to evaluate the same trade in parallel
    for _ in range(20):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify that exactly one thread successfully authorized the intent
    authorized = [r for r in results if r.status == ExecutionAuthorizationStatus.AUTHORIZED]
    rejected = [r for r in results if r.status == ExecutionAuthorizationStatus.REJECTED]

    assert len(authorized) == 1
    assert len(rejected) == 19
    for r in rejected:
        assert "Duplicate execution request" in r.rejection_reason or "DUPLICATE_INTENT" in r.triggered_rules


# 11. Static AST Architecture Isolation check
def test_execution_authorization_import_isolation():
    """
    Ensures that backend/execution_authorization/ package has zero dependencies on:
    - ccxt, binance, bybit, base_ai_provider, ollama_provider, PaperExecutor, http requests.
    """
    package_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "backend", "execution_authorization")
    )
    forbidden = ["ccxt", "binance", "bybit", "urllib", "requests", "aiohttp", "PaperExecutor", "OllamaProvider", "ReasoningEngine"]

    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            for bad in forbidden:
                                assert bad not in name.name, f"Forbidden import '{name.name}' in {path}"
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for bad in forbidden:
                                assert bad not in node.module, f"Forbidden import from '{node.module}' in {path}"
