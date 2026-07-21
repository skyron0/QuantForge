import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any
from dataclasses import replace

from backend.portfolio.models import PositionSide
from backend.execution_authorization.models import OrderDirection
from backend.position_lifecycle.exceptions import (
    PositionNotFoundError,
    PositionStateError,
    StaleMarketDataError,
    LifecycleInvariantError,
    DuplicateTriggerError
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
from backend.position_lifecycle.protective import (
    validate_protective_levels,
    update_trailing_and_check_triggers
)
from backend.position_lifecycle.telemetry import PositionLifecycleTelemetrySink


def parse_iso(ts: str) -> datetime:
    """Helper to parse ISO strings, replacing Z with UTC timezone offset."""
    if not ts:
        raise ValueError("Empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {ts}") from e


class PositionLifecycleEngine:
    def __init__(
        self,
        policy: PositionLifecyclePolicy,
        store: PositionLifecycleStore,
        telemetry_sink: Optional[PositionLifecycleTelemetrySink] = None
    ):
        self.policy = policy
        self.store = store
        self.telemetry_sink = telemetry_sink

    def register_position(
        self,
        position_id: str,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        average_entry_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
        trailing_stop_enabled: bool,
        trailing_distance: Optional[Decimal],
        trailing_activation_price: Optional[Decimal],
        timestamp: str,
        source_proposal_id: Optional[str] = None,
        source_execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Registers a new position with protective levels.
        If an open lifecycle already exists for this position:
          - If the side/execution changes (Reversal), close the old lifecycle first and open a new one.
          - Otherwise, raise an error or update. Here, we transition the old one to CLOSED to preserve history.
        """
        # Validate levels first
        validate_protective_levels(
            side=side,
            entry_price=average_entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_enabled=trailing_stop_enabled,
            trailing_distance=trailing_distance,
            trailing_activation_price=trailing_activation_price,
            policy=self.policy
        )

        lifecycle_id = str(uuid.uuid4())
        
        # Check existing active state
        existing = self.store.get(position_id)
        if existing and existing.status != PositionLifecycleStatus.CLOSED:
            # Reversal: close the old one to preserve history
            self.store.close(position_id, timestamp)

        new_state = ProtectivePositionState(
            lifecycle_id=lifecycle_id,
            position_id=position_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            average_entry_price=average_entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_enabled=trailing_stop_enabled,
            trailing_distance=trailing_distance,
            trailing_activation_price=trailing_activation_price,
            highest_price_since_entry=average_entry_price,
            lowest_price_since_entry=average_entry_price,
            active_trailing_stop_price=None,
            status=PositionLifecycleStatus.OPEN,
            created_at=timestamp,
            updated_at=timestamp,
            policy_version=self.policy.policy_version,
            source_proposal_id=source_proposal_id,
            source_execution_id=source_execution_id,
            metadata=metadata or {}
        )
        
        self.store.put(new_state)
        return lifecycle_id

    def evaluate(
        self,
        position_id: str,
        market_price: Decimal,
        market_timestamp: str,
        system_timestamp: str
    ) -> Optional[ExitProposal]:
        """
        Evaluates a market price update against the position's protective levels.
        Updates trailing state and evaluates triggers.
        Returns an ExitProposal if a trigger is hit.
        """
        start_counter = time.perf_counter()
        rejection_reason: Optional[str] = None
        
        # Fetch current state
        state = self.store.get(position_id)
        if not state:
            raise PositionNotFoundError(f"Position {position_id} not found in store")

        if state.status == PositionLifecycleStatus.CLOSED:
            # A closed lifecycle cannot trigger
            return None

        try:
            # 1. Market Data Freshness checks
            mkt_dt = parse_iso(market_timestamp)
            sys_dt = parse_iso(system_timestamp)

            age = (sys_dt - mkt_dt).total_seconds()
            if age > self.policy.maximum_market_data_age_seconds:
                rejection_reason = f"Stale market data: age={age}s, max={self.policy.maximum_market_data_age_seconds}s"
                raise StaleMarketDataError(rejection_reason)

            skew = (mkt_dt - sys_dt).total_seconds()
            if skew > self.policy.maximum_future_clock_skew_seconds:
                rejection_reason = f"Clock skew: skew={skew}s, max={self.policy.maximum_future_clock_skew_seconds}s"
                raise StaleMarketDataError(rejection_reason)

            # If the status is CLOSING, it represents that an exit has already been triggered.
            # We must not generate another independent ExitProposal.
            if state.status == PositionLifecycleStatus.CLOSING:
                rejection_reason = "Position already in CLOSING state"
                return None

            # 2. Trigger Evaluation and Trailing Stop Update
            result = update_trailing_and_check_triggers(
                state=state,
                market_price=market_price,
                policy=self.policy
            )
            triggered_type, trigger_price, new_high, new_low, new_trailing, new_stop_loss = result

            # Form updated state
            updated_state = replace(
                state,
                highest_price_since_entry=new_high,
                lowest_price_since_entry=new_low,
                active_trailing_stop_price=new_trailing,
                stop_loss=new_stop_loss,
                updated_at=system_timestamp
            )

            proposal: Optional[ExitProposal] = None
            if triggered_type:
                # Transition status to CLOSING
                updated_state = replace(updated_state, status=PositionLifecycleStatus.CLOSING)
                
                # Determine exit details
                direction = OrderDirection.SELL if state.side == PositionSide.LONG else OrderDirection.BUY
                
                reason_map = {
                    ProtectiveTriggerType.STOP_LOSS: ExitReason.STOP_LOSS_TRIGGERED,
                    ProtectiveTriggerType.TAKE_PROFIT: ExitReason.TAKE_PROFIT_TRIGGERED,
                    ProtectiveTriggerType.TRAILING_STOP: ExitReason.TRAILING_STOP_TRIGGERED,
                    ProtectiveTriggerType.MANUAL_EXIT: ExitReason.MANUAL_EXIT
                }
                exit_reason = reason_map.get(triggered_type, ExitReason.RISK_EXIT)
                
                # Convert timestamps to float for TTL
                sys_unix = sys_dt.timestamp()
                expiry_unix = sys_unix + self.policy.exit_proposal_ttl_seconds
                expires_at_str = datetime.fromtimestamp(expiry_unix, timezone.utc).isoformat()

                proposal = ExitProposal(
                    exit_proposal_id=str(uuid.uuid4()),
                    lifecycle_id=state.lifecycle_id,
                    position_id=state.position_id,
                    symbol=state.symbol,
                    position_side=state.side,
                    exit_direction=direction,
                    requested_quantity=state.quantity,
                    trigger_type=triggered_type,
                    exit_reason=exit_reason,
                    trigger_price=trigger_price,
                    market_price=market_price,
                    source_stop_loss=state.stop_loss,
                    source_take_profit=state.take_profit,
                    source_trailing_stop=state.active_trailing_stop_price,
                    created_at=system_timestamp,
                    expires_at=expires_at_str,
                    lifecycle_policy_version=self.policy.policy_version,
                    source_execution_id=state.source_execution_id
                )

            # Save state update
            self.store.update(updated_state)
            
            # Telemetry Sink logging
            if self.telemetry_sink:
                latency_ms = (time.perf_counter() - start_counter) * 1000.0
                self.telemetry_sink.record_evaluation(
                    lifecycle_id=state.lifecycle_id,
                    position_id=state.position_id,
                    symbol=state.symbol,
                    position_side=state.side,
                    market_price=float(market_price),
                    stop_loss=float(updated_state.stop_loss) if updated_state.stop_loss else None,
                    take_profit=float(updated_state.take_profit) if updated_state.take_profit else None,
                    active_trailing_stop=float(updated_state.active_trailing_stop_price) if updated_state.active_trailing_stop_price else None,
                    highest_price_since_entry=float(updated_state.highest_price_since_entry) if updated_state.highest_price_since_entry else None,
                    lowest_price_since_entry=float(updated_state.lowest_price_since_entry) if updated_state.lowest_price_since_entry else None,
                    trigger_type=triggered_type,
                    exit_proposal_generated=(proposal is not None),
                    lifecycle_status=updated_state.status,
                    policy_version=self.policy.policy_version,
                    latency_ms=latency_ms,
                    rejection_reason=None
                )
                
            return proposal

        except Exception as e:
            # Telemetry Sink error logging
            if self.telemetry_sink:
                latency_ms = (time.perf_counter() - start_counter) * 1000.0
                self.telemetry_sink.record_evaluation(
                    lifecycle_id=state.lifecycle_id,
                    position_id=state.position_id,
                    symbol=state.symbol,
                    position_side=state.side,
                    market_price=float(market_price),
                    stop_loss=float(state.stop_loss) if state.stop_loss else None,
                    take_profit=float(state.take_profit) if state.take_profit else None,
                    active_trailing_stop=float(state.active_trailing_stop_price) if state.active_trailing_stop_price else None,
                    highest_price_since_entry=float(state.highest_price_since_entry) if state.highest_price_since_entry else None,
                    lowest_price_since_entry=float(state.lowest_price_since_entry) if state.lowest_price_since_entry else None,
                    trigger_type=None,
                    exit_proposal_generated=False,
                    lifecycle_status=state.status,
                    policy_version=self.policy.policy_version,
                    latency_ms=latency_ms,
                    rejection_reason=str(e)
                )
            raise

    def synchronize_position(
        self,
        position_id: str,
        current_quantity: Decimal,
        timestamp: str
    ) -> None:
        """
        Synchronizes the protective state quantity when a position changes (due to partial fills).
        If quantity falls to zero, closes the position lifecycle.
        """
        if current_quantity.is_nan() or current_quantity.is_infinite() or current_quantity < Decimal("0"):
            raise ValueError(f"Invalid quantity to synchronize: {current_quantity}")

        state = self.store.get(position_id)
        if not state:
            return  # No active protective monitoring for this position

        if state.status == PositionLifecycleStatus.CLOSED:
            return

        if current_quantity == Decimal("0"):
            # Fully closed
            self.store.close(position_id, timestamp)
        else:
            # Partial close - update quantity
            updated_state = replace(
                state,
                quantity=current_quantity,
                updated_at=timestamp
            )
            self.store.update(updated_state)
