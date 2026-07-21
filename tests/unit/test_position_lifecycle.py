import ast
import os
import time
import pytest
import threading
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.portfolio.models import PositionSide
from backend.execution_authorization.models import OrderDirection, ExecutionContext, ExecutionEnvironment, ExecutionAuthorizationStatus
from backend.execution_authorization.policy import ExecutionPolicy

from backend.position_lifecycle.exceptions import (
    PositionLifecycleValidationError,
    InvalidLifecyclePolicyError,
    InvalidStopLossError,
    InvalidTakeProfitError,
    InvalidTrailingStopError,
    PositionStateError,
    PositionNotFoundError,
    StaleMarketDataError
)
from backend.position_lifecycle.models import (
    PositionLifecycleStatus,
    ProtectiveTriggerType,
    ExitReason,
    ProtectivePositionState,
    ExitProposal
)
from backend.position_lifecycle.policy import PositionLifecyclePolicy
from backend.position_lifecycle.store import PositionLifecycleStore
from backend.position_lifecycle.lifecycle import PositionLifecycleEngine
from backend.position_lifecycle.bridge import ExitExecutionRequestBuilder, ExitAuthorizationEngine
from backend.position_lifecycle.telemetry import PositionLifecycleTelemetrySink


from typing import Dict, Any

class MockTelemetrySink(PositionLifecycleTelemetrySink):
    def __init__(self):
        self.records = []

    def record_evaluation(
        self,
        lifecycle_id: str,
        position_id: str,
        symbol: str,
        position_side: PositionSide,
        market_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        active_trailing_stop: Optional[float],
        highest_price_since_entry: Optional[float],
        lowest_price_since_entry: Optional[float],
        trigger_type: Optional[ProtectiveTriggerType],
        exit_proposal_generated: bool,
        lifecycle_status: PositionLifecycleStatus,
        policy_version: str,
        latency_ms: float,
        rejection_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.records.append({
            "lifecycle_id": lifecycle_id,
            "position_id": position_id,
            "symbol": symbol,
            "position_side": position_side,
            "market_price": market_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "active_trailing_stop": active_trailing_stop,
            "highest_price_since_entry": highest_price_since_entry,
            "lowest_price_since_entry": lowest_price_since_entry,
            "trigger_type": trigger_type,
            "exit_proposal_generated": exit_proposal_generated,
            "lifecycle_status": lifecycle_status,
            "policy_version": policy_version,
            "latency_ms": latency_ms,
            "rejection_reason": rejection_reason,
            "metadata": metadata
        })


@pytest.fixture
def base_policy():
    return PositionLifecyclePolicy(
        policy_version="1.0.0",
        allow_stop_loss=True,
        require_stop_loss=False,
        allow_take_profit=True,
        require_take_profit=False,
        allow_trailing_stop=True,
        minimum_stop_distance_fraction=Decimal("0.01"),
        maximum_stop_distance_fraction=Decimal("0.20"),
        minimum_take_profit_distance_fraction=Decimal("0.01"),
        trailing_distance_mode="ABSOLUTE",
        minimum_trailing_distance=Decimal("1.0"),
        maximum_trailing_distance=Decimal("100.0"),
        allow_breakeven=True,
        breakeven_activation_fraction=Decimal("0.05"),
        breakeven_offset_fraction=Decimal("0.005")
    )


@pytest.fixture
def empty_store():
    return PositionLifecycleStore()


@pytest.fixture
def mock_telemetry():
    return MockTelemetrySink()


@pytest.fixture
def engine(base_policy, empty_store, mock_telemetry):
    return PositionLifecycleEngine(
        policy=base_policy,
        store=empty_store,
        telemetry_sink=mock_telemetry
    )


# ==================================================
# 1. STOP LOSS TESTS
# ==================================================

def test_stop_loss_validation(engine, base_policy):
    # Valid Long Stop Loss
    engine.register_position(
        position_id="pos_1",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )
    
    # Valid Short Stop Loss
    engine.register_position(
        position_id="pos_2",
        symbol="BTCUSD",
        side=PositionSide.SHORT,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("105.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Invalid Long Stop Loss (stop_loss >= entry_price)
    with pytest.raises(InvalidStopLossError, match="must be strictly below"):
        engine.register_position(
            position_id="pos_3",
            symbol="BTCUSD",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            average_entry_price=Decimal("100.0"),
            stop_loss=Decimal("100.5"),
            take_profit=None,
            trailing_stop_enabled=False,
            trailing_distance=None,
            trailing_activation_price=None,
            timestamp="2026-07-21T12:00:00+00:00"
        )

    # Invalid Short Stop Loss (stop_loss <= entry_price)
    with pytest.raises(InvalidStopLossError, match="must be strictly above"):
        engine.register_position(
            position_id="pos_4",
            symbol="BTCUSD",
            side=PositionSide.SHORT,
            quantity=Decimal("1.0"),
            average_entry_price=Decimal("100.0"),
            stop_loss=Decimal("99.5"),
            take_profit=None,
            trailing_stop_enabled=False,
            trailing_distance=None,
            trailing_activation_price=None,
            timestamp="2026-07-21T12:00:00+00:00"
        )


def test_stop_loss_triggering(engine):
    # LONG Stop Loss trigger
    engine.register_position(
        position_id="pos_long",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.5"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Price stays above SL (no trigger)
    proposal = engine.evaluate(
        position_id="pos_long",
        market_price=Decimal("98.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    assert proposal is None

    # Price hits SL (triggered)
    proposal = engine.evaluate(
        position_id="pos_long",
        market_price=Decimal("94.5"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    assert proposal is not None
    assert proposal.trigger_type == ProtectiveTriggerType.STOP_LOSS
    assert proposal.exit_reason == ExitReason.STOP_LOSS_TRIGGERED
    assert proposal.requested_quantity == Decimal("1.5")
    assert proposal.exit_direction == OrderDirection.SELL

    # SHORT Stop Loss trigger
    engine.register_position(
        position_id="pos_short",
        symbol="BTCUSD",
        side=PositionSide.SHORT,
        quantity=Decimal("2.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("105.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Price hits SL (triggered)
    proposal = engine.evaluate(
        position_id="pos_short",
        market_price=Decimal("106.0"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    assert proposal is not None
    assert proposal.trigger_type == ProtectiveTriggerType.STOP_LOSS
    assert proposal.exit_reason == ExitReason.STOP_LOSS_TRIGGERED
    assert proposal.requested_quantity == Decimal("2.0")
    assert proposal.exit_direction == OrderDirection.BUY


# ==================================================
# 2. TAKE PROFIT TESTS
# ==================================================

def test_take_profit_validation(engine):
    # Valid Long Take Profit
    engine.register_position(
        position_id="pos_1",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=None,
        take_profit=Decimal("110.0"),
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Invalid Long Take Profit
    with pytest.raises(InvalidTakeProfitError, match="must be strictly above"):
        engine.register_position(
            position_id="pos_2",
            symbol="BTCUSD",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            average_entry_price=Decimal("100.0"),
            stop_loss=None,
            take_profit=Decimal("95.0"),
            trailing_stop_enabled=False,
            trailing_distance=None,
            trailing_activation_price=None,
            timestamp="2026-07-21T12:00:00+00:00"
        )


def test_take_profit_triggering(engine):
    # LONG Take Profit trigger
    engine.register_position(
        position_id="pos_long",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=None,
        take_profit=Decimal("110.0"),
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    proposal = engine.evaluate(
        position_id="pos_long",
        market_price=Decimal("108.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    assert proposal is None

    proposal = engine.evaluate(
        position_id="pos_long",
        market_price=Decimal("111.0"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    assert proposal is not None
    assert proposal.trigger_type == ProtectiveTriggerType.TAKE_PROFIT


# ==================================================
# 3. TRAILING STOP TESTS
# ==================================================

def test_trailing_stop_activation_and_adjustment(engine):
    # LONG position, trailing activated immediately (no activation threshold)
    engine.register_position(
        position_id="pos_trail",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=None,
        take_profit=None,
        trailing_stop_enabled=True,
        trailing_distance=Decimal("5.0"),
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Initial evaluation: high updates to 102.0. stop moves to 102 - 5 = 97.
    engine.evaluate(
        position_id="pos_trail",
        market_price=Decimal("102.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    state = engine.store.get("pos_trail")
    assert state.active_trailing_stop_price == Decimal("97.0")

    # Price moves up to 108.0, stop moves up to 108 - 5 = 103.
    engine.evaluate(
        position_id="pos_trail",
        market_price=Decimal("108.0"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    state = engine.store.get("pos_trail")
    assert state.active_trailing_stop_price == Decimal("103.0")

    # Price moves down to 106.0, stop MUST NOT move down (remains 103).
    engine.evaluate(
        position_id="pos_trail",
        market_price=Decimal("106.0"),
        market_timestamp="2026-07-21T12:00:15+00:00",
        system_timestamp="2026-07-21T12:00:15+00:00"
    )
    state = engine.store.get("pos_trail")
    assert state.active_trailing_stop_price == Decimal("103.0")

    # Price hits the trailing stop (103.0), triggers ExitProposal
    proposal = engine.evaluate(
        position_id="pos_trail",
        market_price=Decimal("102.5"),
        market_timestamp="2026-07-21T12:00:20+00:00",
        system_timestamp="2026-07-21T12:00:20+00:00"
    )
    assert proposal is not None
    assert proposal.trigger_type == ProtectiveTriggerType.TRAILING_STOP
    assert proposal.exit_reason == ExitReason.TRAILING_STOP_TRIGGERED


def test_trailing_activation_threshold(engine):
    # LONG position, trailing activated only at 105.0
    engine.register_position(
        position_id="pos_activate",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=None,
        take_profit=None,
        trailing_stop_enabled=True,
        trailing_distance=Decimal("5.0"),
        trailing_activation_price=Decimal("105.0"),
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Price is 103.0. Activation threshold not reached, active trailing stop remains None
    engine.evaluate(
        position_id="pos_activate",
        market_price=Decimal("103.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    state = engine.store.get("pos_activate")
    assert state.active_trailing_stop_price is None

    # Price is 106.0. Activation threshold reached! active stop becomes 106 - 5 = 101.
    engine.evaluate(
        position_id="pos_activate",
        market_price=Decimal("106.0"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    state = engine.store.get("pos_activate")
    assert state.active_trailing_stop_price == Decimal("101.0")


# ==================================================
# 4. BREAKEVEN TESTS
# ==================================================

def test_breakeven_adjustment(engine):
    # LONG position, entry=100.0, SL=94.0. Breakeven activates at +5% (105.0), offsets SL to entry+0.5% (100.5)
    engine.register_position(
        position_id="pos_be",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("94.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Price moves to 103.0. SL remains 94.0
    engine.evaluate(
        position_id="pos_be",
        market_price=Decimal("103.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    state = engine.store.get("pos_be")
    assert state.stop_loss == Decimal("94.0")

    # Price moves to 105.1. Breakeven activates! SL adjusts to 100.5
    engine.evaluate(
        position_id="pos_be",
        market_price=Decimal("105.1"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    state = engine.store.get("pos_be")
    assert state.stop_loss == Decimal("100.5")

    # SL must not adjust back down if price decreases
    engine.evaluate(
        position_id="pos_be",
        market_price=Decimal("102.0"),
        market_timestamp="2026-07-21T12:00:15+00:00",
        system_timestamp="2026-07-21T12:00:15+00:00"
    )
    state = engine.store.get("pos_be")
    assert state.stop_loss == Decimal("100.5")


# ==================================================
# 5. LIFECYCLE MACHINE & WORKSPACE & REVERSAL
# ==================================================

def test_machine_states_and_idempotency(engine):
    engine.register_position(
        position_id="pos_lfc",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Initial state is OPEN
    assert engine.store.get("pos_lfc").status == PositionLifecycleStatus.OPEN

    # Evaluates and triggers SL: transitions state to CLOSING
    proposal = engine.evaluate(
        position_id="pos_lfc",
        market_price=Decimal("94.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    assert proposal is not None
    assert engine.store.get("pos_lfc").status == PositionLifecycleStatus.CLOSING

    # Evaluate again while in CLOSING does not output another ExitProposal
    proposal2 = engine.evaluate(
        position_id="pos_lfc",
        market_price=Decimal("93.0"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    assert proposal2 is None


def test_partial_quantity_synchronization(engine):
    engine.register_position(
        position_id="pos_sync",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Sync partial remaining: quantity becomes 0.4
    engine.synchronize_position(
        position_id="pos_sync",
        current_quantity=Decimal("0.4"),
        timestamp="2026-07-21T12:05:00+00:00"
    )
    state = engine.store.get("pos_sync")
    assert state.quantity == Decimal("0.4")
    assert state.status == PositionLifecycleStatus.OPEN

    # Sync full close: transitions status to CLOSED
    engine.synchronize_position(
        position_id="pos_sync",
        current_quantity=Decimal("0.0"),
        timestamp="2026-07-21T12:06:00+00:00"
    )
    state = engine.store.get("pos_sync")
    assert state.status == PositionLifecycleStatus.CLOSED


def test_reversal_creates_new_lifecycle(engine):
    # LONG Position
    engine.register_position(
        position_id="pos_rev",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )
    lfc1 = engine.store.get("pos_rev")
    assert lfc1.side == PositionSide.LONG
    assert lfc1.status == PositionLifecycleStatus.OPEN

    # Reversal to SHORT
    engine.register_position(
        position_id="pos_rev",
        symbol="BTCUSD",
        side=PositionSide.SHORT,
        quantity=Decimal("1.5"),
        average_entry_price=Decimal("101.0"),
        stop_loss=Decimal("106.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:01:00+00:00"
    )

    # Verify old lifecycle is now CLOSED in audit store (or retrieve by lifecycle_id)
    old_state = engine.store.get_by_lifecycle_id(lfc1.lifecycle_id)
    assert old_state.status == PositionLifecycleStatus.CLOSED

    # New state is independent and active
    new_state = engine.store.get("pos_rev")
    assert new_state.lifecycle_id != lfc1.lifecycle_id
    assert new_state.side == PositionSide.SHORT
    assert new_state.status == PositionLifecycleStatus.OPEN


# ==================================================
# 6. MARKET SAFETY & VALS
# ==================================================

def test_market_data_freshness_validations(engine):
    engine.register_position(
        position_id="pos_safety",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # 1. Stale market data age exceeds maximum_market_data_age_seconds
    with pytest.raises(StaleMarketDataError, match="Stale market data"):
        engine.evaluate(
            position_id="pos_safety",
            market_price=Decimal("94.0"),
            market_timestamp="2026-07-21T11:58:00+00:00",  # 2 minutes stale
            system_timestamp="2026-07-21T12:00:00+00:00"
        )

    # 2. Clock skew future drift exceeds maximum_future_clock_skew_seconds
    with pytest.raises(StaleMarketDataError, match="Clock skew"):
        engine.evaluate(
            position_id="pos_safety",
            market_price=Decimal("94.0"),
            market_timestamp="2026-07-21T12:01:00+00:00",  # future skew
            system_timestamp="2026-07-21T12:00:00+00:00"
        )


def test_nan_inf_safety(engine):
    with pytest.raises(ValueError):
        engine.register_position(
            position_id="pos_nan",
            symbol="BTCUSD",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            average_entry_price=Decimal("NaN"),
            stop_loss=Decimal("95.0"),
            take_profit=None,
            trailing_stop_enabled=False,
            trailing_distance=None,
            trailing_activation_price=None,
            timestamp="2026-07-21T12:00:00+00:00"
        )

    engine.register_position(
        position_id="pos_nan_eval",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    with pytest.raises(ValueError):
        # Evaluation must fail-closed on Inf or NaN values
        engine.evaluate(
            position_id="pos_nan_eval",
            market_price=Decimal("Infinity"),
            market_timestamp="2026-07-21T12:00:05+00:00",
            system_timestamp="2026-07-21T12:00:05+00:00"
        )


# ==================================================
# 7. GAP & MULTIPLE TRIGGER PRIORITY TESTS
# ==================================================

def test_gap_limitations_and_priority(engine):
    # LONG position: SL=95, TP=110
    engine.register_position(
        position_id="pos_gap",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=Decimal("110.0"),
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Extreme gap update: market price becomes NaN/0/negative
    with pytest.raises(ValueError):
        engine.evaluate(
            position_id="pos_gap",
            market_price=Decimal("-10.0"),
            market_timestamp="2026-07-21T12:00:05+00:00",
            system_timestamp="2026-07-21T12:00:05+00:00"
        )


    # Extreme gap crosses both SL and TP (e.g. price sequence jumps).
    # Since trigger_priority is [STOP_LOSS, TAKE_PROFIT], SL takes precedence
    # To test this, we mock/set both triggers to be crossed. Wait, how can both be crossed?
    # If take_profit is 94 and stop_loss is 96? No, they have to be on opposite sides of entry.
    # But wait, what if we have a gapped observation that is extremely gapped?
    # Actually, a single price cannot be both <= 95 and >= 110.
    # But wait, we can test this if we configure priority on trailing stops and stops, which are on the same side.
    # For example, if stop_loss = 95.0, and trailing stop is activated at 98.0.
    # Let's say stop_loss = 95.0 and active_trailing_stop_price = 97.0.
    # If price drops to 94.0, both STOP_LOSS (95) and TRAILING_STOP (97) are triggered.
    # By default, trigger_priority has STOP_LOSS before TRAILING_STOP, so SL is triggered.
    # Let's verify this!
    engine.register_position(
        position_id="pos_dual",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=True,
        trailing_distance=Decimal("3.0"),  # active trail stop will be 97.0
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    # Touch 100 to set trailing stop high to 100. Active trailing stop becomes 97
    engine.evaluate(
        position_id="pos_dual",
        market_price=Decimal("100.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )
    
    # Drop to 94.0. Both SL (<=95) and Trailing stop (<=97) are triggered.
    # Policy lists STOP_LOSS first.
    proposal = engine.evaluate(
        position_id="pos_dual",
        market_price=Decimal("94.0"),
        market_timestamp="2026-07-21T12:00:10+00:00",
        system_timestamp="2026-07-21T12:00:10+00:00"
    )
    assert proposal is not None
    assert proposal.trigger_type == ProtectiveTriggerType.STOP_LOSS
    assert proposal.exit_reason == ExitReason.STOP_LOSS_TRIGGERED


# ==================================================
# 8. BRIDGE & EXIT SECURITY BOUNDARY
# ==================================================

def test_exit_execution_request_builder_and_bridge(base_policy):
    proposal = ExitProposal(
        exit_proposal_id="prop_exit_123",
        lifecycle_id="lfc_123",
        position_id="pos_123",
        symbol="BTCUSD",
        position_side=PositionSide.LONG,
        exit_direction=OrderDirection.SELL,
        requested_quantity=Decimal("1.5"),
        trigger_type=ProtectiveTriggerType.STOP_LOSS,
        exit_reason=ExitReason.STOP_LOSS_TRIGGERED,
        trigger_price=Decimal("95.0"),
        market_price=Decimal("94.0"),
        source_stop_loss=Decimal("95.0"),
        source_take_profit=None,
        source_trailing_stop=None,
        created_at="2026-07-21T12:00:00+00:00",
        expires_at="2026-07-21T12:01:00+00:00",
        lifecycle_policy_version="1.0.0",
        source_execution_id="exec_123"
    )

    from backend.execution_authorization.models import OrderType

    exec_policy = ExecutionPolicy(
        policy_version="1.0.0",
        allowed_environments=[ExecutionEnvironment.PAPER],
        maximum_market_data_age_seconds=60.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.01,
        maximum_quantity=1000.0,
        require_stop_loss=False,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET],
        allow_live_execution_intents=False,
        require_execution_enabled=True,
        reject_when_kill_switch_active=True,
        require_symbol_enabled=True,
        maximum_clock_skew_seconds=5.0
    )

    context = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp="2026-07-21T12:00:05+00:00",
        market_timestamp="2026-07-21T12:00:05+00:00",
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=94.0,
        metadata={"symbol": "BTCUSD"}
    )

    auth_engine = ExitAuthorizationEngine(exec_policy)
    auth_result = auth_engine.authorize_exit(
        proposal=proposal,
        context=context,
        risk_policy_version="risk_p_v1",
        sizing_policy_version="size_p_v1",
        idempotency_key="idemp_key_123"
    )

    assert auth_result.status == ExecutionAuthorizationStatus.AUTHORIZED
    assert auth_result.intent is not None
    assert auth_result.intent.symbol == "BTCUSD"
    assert auth_result.intent.quantity == 1.5
    assert auth_result.intent.direction == OrderDirection.SELL
    assert auth_result.intent.idempotency_key == "idemp_key_123"

    # Expired Exit Proposal rejection
    late_context = ExecutionContext(
        environment=ExecutionEnvironment.PAPER,
        current_timestamp="2026-07-21T12:02:00+00:00",  # Expired
        market_timestamp="2026-07-21T12:00:05+00:00",
        execution_enabled=True,
        kill_switch_active=False,
        symbol_trading_enabled=True,
        available_balance=10000.0,
        current_price=94.0,
        metadata={"symbol": "BTCUSD"}
    )
    auth_result_rej = auth_engine.authorize_exit(
        proposal=proposal,
        context=late_context,
        risk_policy_version="risk_p_v1",
        sizing_policy_version="size_p_v1",
        idempotency_key="idemp_key_123"
    )
    assert auth_result_rej.status == ExecutionAuthorizationStatus.REJECTED


# ==================================================
# 9. CONCURRENCY & THREAD SAFETY
# ==================================================

def test_thread_safety_and_single_exit_generation(engine):
    engine.register_position(
        position_id="pos_concur",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    results = []
    def worker():
        try:
            prop = engine.evaluate(
                position_id="pos_concur",
                market_price=Decimal("94.0"),
                market_timestamp="2026-07-21T12:00:05+00:00",
                system_timestamp="2026-07-21T12:00:05+00:00"
            )
            if prop is not None:
                results.append(prop)
        except Exception:
            pass

    threads = []
    for _ in range(10):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Only one thread should have generated an ExitProposal
    assert len(results) == 1


# ==================================================
# 10. ARCHITECTURAL ISOLATION AST CHECKS
# ==================================================

def test_package_dependencies_isolation():
    """
    AST analysis to verify that backend/position_lifecycle has ZERO
    dependencies on CCXT, OllamaProvider, broker clients, or direct execution.
    """
    package_dir = os.path.join("backend", "position_lifecycle")
    forbidden_imports = {
        "ccxt", "binance", "bybit", "ollama", "PaperExecutionAdapter", 
        "requests", "urllib", "http", "socket"
    }

    for root, _, files in os.walk(package_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                # Check import name
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base_module = alias.name.split(".")[0]
                        assert base_module not in forbidden_imports, (
                            f"Forbidden import '{alias.name}' detected in {filepath}"
                        )
                # Check import from
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_module = node.module.split(".")[0]
                        assert base_module not in forbidden_imports, (
                            f"Forbidden import from '{node.module}' detected in {filepath}"
                        )
                        # Check specific names imported
                        for alias in node.names:
                            assert alias.name not in forbidden_imports, (
                                f"Forbidden imported symbol '{alias.name}' in module '{node.module}' in {filepath}"
                            )


def test_telemetry_latency_recording(engine, mock_telemetry):
    engine.register_position(
        position_id="pos_tel",
        symbol="BTCUSD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        stop_loss=Decimal("95.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp="2026-07-21T12:00:00+00:00"
    )

    engine.evaluate(
        position_id="pos_tel",
        market_price=Decimal("99.0"),
        market_timestamp="2026-07-21T12:00:05+00:00",
        system_timestamp="2026-07-21T12:00:05+00:00"
    )

    assert len(mock_telemetry.records) == 1
    assert "latency_ms" in mock_telemetry.records[0]
    assert mock_telemetry.records[0]["latency_ms"] >= 0.0
