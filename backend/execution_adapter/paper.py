import math
import uuid
import time
from typing import List, Optional, Callable
from datetime import datetime, timezone

from backend.replay.clock import Clock, SystemClock

from backend.execution_authorization.models import OrderIntent, ExecutionEnvironment, OrderDirection, OrderType
from backend.execution_adapter.exceptions import (
    ExecutionAdapterValidationError,
    UnsupportedExecutionEnvironmentError,
    UnsupportedOrderTypeError,
    StaleExecutionContextError,
    InsufficientLiquidityError,
    DuplicateExecutionError,
    InvalidMarketStateError,
    ExecutionSimulationError
)
from backend.execution_adapter.models import (
    ExecutionStatus,
    Fill,
    ExecutionResult,
    PaperExecutionContext
)
from backend.execution_adapter.policy import PaperExecutionPolicy
from backend.execution_adapter.idempotency import ExecutionIdempotencyStore
from backend.execution_adapter.telemetry import ExecutionTelemetrySink, ConsoleExecutionTelemetrySink

from backend.execution_adapter.base import BaseExecutionAdapter

def parse_iso(iso_str: str) -> datetime:
    try:
        # Standard timezone parsing or replace Z with +00:00
        normalized = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except Exception as e:
        raise ExecutionAdapterValidationError(f"Invalid timestamp format: {iso_str}") from e

class PaperExecutionAdapter(BaseExecutionAdapter):
    def __init__(
        self,
        policy: PaperExecutionPolicy,
        idempotency_store: Optional[ExecutionIdempotencyStore] = None,
        telemetry_sink: Optional[ExecutionTelemetrySink] = None,
        adapter_name: str = "PaperExecutionAdapter",
        clock: Optional[Clock] = None,
        uuid_generator: Optional[Callable[[], str]] = None
    ):
        self.policy = policy
        self.idempotency_store = idempotency_store or ExecutionIdempotencyStore(
            ttl_seconds=policy.execution_result_ttl_seconds
        )
        self.telemetry_sink = telemetry_sink or ConsoleExecutionTelemetrySink()
        self.adapter_name = adapter_name
        self.clock = clock or SystemClock()
        self._uuid_generator = uuid_generator or (lambda: str(uuid.uuid4()))

    def execute(self, intent: OrderIntent, context: PaperExecutionContext) -> ExecutionResult:
        start_counter = time.perf_counter()
        
        # 1. Parameter Checks (Fail-Closed on None)
        if intent is None or context is None:
            raise ExecutionAdapterValidationError("Order intent and market context must be provided")

        # 2. Duplicate Execution Prevention Check
        # Attempt atomic claim. If it fails, raise DuplicateExecutionError.
        if not self.idempotency_store.claim(intent.intent_id):
            # Check if this execution is already completed
            res = self.idempotency_store.get_result(intent.intent_id)
            if res is not None:
                return res
            raise DuplicateExecutionError(f"Execution for intent {intent.intent_id} is already in progress or completed")

        try:
            # 3. Environment Isolation Opt-In
            if intent.environment != ExecutionEnvironment.PAPER:
                raise UnsupportedExecutionEnvironmentError(
                    f"Paper adapter only supports PAPER environment. Attempted: {intent.environment.value}"
                )

            # 4. Basic Intent Validation
            if not intent.intent_id or not intent.proposal_id or not intent.risk_authorization_id or not intent.sizing_id:
                raise ExecutionAdapterValidationError("Intent is missing required lineage identifiers")
            
            if not intent.symbol:
                raise ExecutionAdapterValidationError("Intent is missing trading symbol")

            if math.isnan(intent.quantity) or math.isinf(intent.quantity) or intent.quantity <= 0:
                raise ExecutionAdapterValidationError(f"Invalid quantity in intent: {intent.quantity}")

            # System time comparison for freshness and expiry
            sys_dt = self.clock.now()
            mkt_dt = parse_iso(context.timestamp)
            intent_expiry_dt = parse_iso(intent.expires_at)

            # Future skew check
            if mkt_dt > sys_dt:
                skew = (mkt_dt - sys_dt).total_seconds()
                if skew > self.policy.maximum_future_clock_skew_seconds:
                    raise StaleExecutionContextError(
                        f"Context timestamp is in the future. Skew of {skew}s exceeds policy limit"
                    )
            else:
                # Stale market data check
                age = (sys_dt - mkt_dt).total_seconds()
                if age > self.policy.maximum_market_data_age_seconds:
                    raise StaleExecutionContextError(
                        f"Context timestamp is too old. Age of {age}s exceeds policy limit"
                    )

            # Expiry Check
            if mkt_dt > intent_expiry_dt:
                # Return rejected result for expired intent
                # Note: Rejections don't raise validation exceptions, they return REJECTED status.
                result = self._build_rejected_result(
                    intent, "Order intent has expired", start_counter, mkt_dt
                )
                self.idempotency_store.complete(intent.intent_id, result)
                return result

            # 5. Order Type Support Verification
            if intent.order_type not in (OrderType.MARKET, OrderType.LIMIT):
                raise UnsupportedOrderTypeError(f"Unsupported order type: {intent.order_type.value}")

            # 6. Execute Order Simulation
            # Base price reference
            if intent.direction == OrderDirection.BUY:
                base_price = context.ask_price
            elif intent.direction == OrderDirection.SELL:
                base_price = context.bid_price
            else:
                raise ExecutionAdapterValidationError(f"Unsupported direction: {intent.direction}")

            # LIMIT Condition Check
            if intent.order_type == OrderType.LIMIT:
                limit_val = intent.limit_price
                if limit_val is None:
                    raise ExecutionAdapterValidationError("LIMIT order requires a valid limit price")
                if math.isnan(limit_val) or limit_val <= 0:
                    raise ExecutionAdapterValidationError("LIMIT order requires a valid limit price")

                # Verify limit condition:
                condition_met = False
                if intent.direction == OrderDirection.BUY:
                    if context.ask_price <= limit_val:
                        condition_met = True
                elif intent.direction == OrderDirection.SELL:
                    if context.bid_price >= limit_val:
                        condition_met = True

                if not condition_met:
                    # ACCEPTED status represents order accepted but condition not yet met to fill
                    result = self._build_accepted_unfilled_result(intent, start_counter, mkt_dt)
                    self.idempotency_store.complete(intent.intent_id, result)
                    return result

            # Quantity and Liquidity sizing check
            requested_qty = intent.quantity
            avail_liq = context.available_liquidity

            if requested_qty > avail_liq:
                if self.policy.reject_if_insufficient_liquidity:
                    result = self._build_rejected_result(
                        intent, "Insufficient available market liquidity", start_counter, mkt_dt
                    )
                    self.idempotency_store.complete(intent.intent_id, result)
                    return result

                if not self.policy.allow_partial_fills:
                    result = self._build_rejected_result(
                        intent, "Insufficient market liquidity and partial fills disabled", start_counter, mkt_dt
                    )
                    self.idempotency_store.complete(intent.intent_id, result)
                    return result

                filled_qty = avail_liq
            else:
                filled_qty = requested_qty

            # Quality Check: Filled quantity must never exceed requested or available liquidity
            if filled_qty > requested_qty:
                filled_qty = requested_qty
            if filled_qty > avail_liq:
                filled_qty = avail_liq

            # Check min quantity bounds
            if filled_qty < self.policy.minimum_fill_quantity:
                result = self._build_rejected_result(
                    intent, f"Filled quantity {filled_qty} is below policy minimum {self.policy.minimum_fill_quantity}", start_counter, mkt_dt
                )
                self.idempotency_store.complete(intent.intent_id, result)
                return result

            # Determine Fill Price, Slippage, and Fees
            if intent.order_type == OrderType.MARKET:
                slippage_rate = self.policy.slippage_rate
                if intent.direction == OrderDirection.BUY:
                    fill_price = base_price * (1.0 + slippage_rate)
                    total_slippage = filled_qty * base_price * slippage_rate
                else:
                    fill_price = base_price * (1.0 - slippage_rate)
                    total_slippage = filled_qty * base_price * slippage_rate
            elif intent.order_type == OrderType.LIMIT:
                limit_val = intent.limit_price
                if limit_val is None:
                    raise ExecutionAdapterValidationError("LIMIT order requires a valid limit price")
                fill_price = limit_val
                total_slippage = 0.0
            else:
                raise UnsupportedOrderTypeError(f"Unsupported order type: {intent.order_type.value}")

            if fill_price <= 0:
                raise ExecutionSimulationError("Calculated fill price is negative or zero")

            filled_notional = filled_qty * fill_price
            total_fees = filled_notional * self.policy.fee_rate

            # Build result
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            
            fill_id = self._uuid_generator()
            fill_timestamp = mkt_dt.isoformat().replace("+00:00", "Z")
            
            fill = Fill(
                fill_id=fill_id,
                intent_id=intent.intent_id,
                symbol=intent.symbol,
                direction=intent.direction,
                quantity=filled_qty,
                price=fill_price,
                notional=filled_notional,
                fee=total_fees,
                slippage_amount=total_slippage,
                timestamp=fill_timestamp,
                metadata={}
            )

            status = ExecutionStatus.FILLED if math.isclose(filled_qty, requested_qty, rel_tol=1e-9) else ExecutionStatus.PARTIALLY_FILLED

            result = ExecutionResult(
                execution_id=self._uuid_generator(),
                intent_id=intent.intent_id,
                proposal_id=intent.proposal_id,
                risk_authorization_id=intent.risk_authorization_id,
                sizing_id=intent.sizing_id,
                symbol=intent.symbol,
                direction=intent.direction,
                requested_quantity=requested_qty,
                filled_quantity=filled_qty,
                average_fill_price=fill_price,
                total_notional=filled_notional,
                total_fees=total_fees,
                total_slippage=total_slippage,
                status=status,
                fills=[fill],
                rejection_reason="",
                adapter_name=self.adapter_name,
                environment=intent.environment,
                started_at=sys_dt.isoformat().replace("+00:00", "Z"),
                completed_at=sys_dt.isoformat().replace("+00:00", "Z"),
                latency_ms=latency_ms,
                policy_version=self.policy.policy_version,
                metadata={}
            )

            # Record telemetry and mark completed
            self.telemetry_sink.record(result, latency_ms)
            self.idempotency_store.complete(intent.intent_id, result)
            return result

        except Exception as e:
            # Release claim on validation/system failure so retry can be attempted
            self.idempotency_store.release(intent.intent_id)
            raise e

    def _build_rejected_result(
        self, intent: OrderIntent, reason: str, start_counter: float, current_dt: datetime
    ) -> ExecutionResult:
        latency_ms = (time.perf_counter() - start_counter) * 1000.0
        ts = current_dt.isoformat().replace("+00:00", "Z")
        result = ExecutionResult(
            execution_id=self._uuid_generator(),
            intent_id=intent.intent_id,
            proposal_id=intent.proposal_id,
            risk_authorization_id=intent.risk_authorization_id,
            sizing_id=intent.sizing_id,
            symbol=intent.symbol,
            direction=intent.direction,
            requested_quantity=intent.quantity,
            filled_quantity=0.0,
            average_fill_price=0.0,
            total_notional=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            status=ExecutionStatus.REJECTED,
            fills=[],
            rejection_reason=reason,
            adapter_name=self.adapter_name,
            environment=intent.environment,
            started_at=ts,
            completed_at=ts,
            latency_ms=latency_ms,
            policy_version=self.policy.policy_version,
            metadata={}
        )
        self.telemetry_sink.record(result, latency_ms)
        return result

    def _build_accepted_unfilled_result(
        self, intent: OrderIntent, start_counter: float, current_dt: datetime
    ) -> ExecutionResult:
        latency_ms = (time.perf_counter() - start_counter) * 1000.0
        ts = current_dt.isoformat().replace("+00:00", "Z")
        result = ExecutionResult(
            execution_id=self._uuid_generator(),
            intent_id=intent.intent_id,
            proposal_id=intent.proposal_id,
            risk_authorization_id=intent.risk_authorization_id,
            sizing_id=intent.sizing_id,
            symbol=intent.symbol,
            direction=intent.direction,
            requested_quantity=intent.quantity,
            filled_quantity=0.0,
            average_fill_price=0.0,
            total_notional=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            status=ExecutionStatus.ACCEPTED,
            fills=[],
            rejection_reason="Limit price condition not met",
            adapter_name=self.adapter_name,
            environment=intent.environment,
            started_at=ts,
            completed_at=ts,
            latency_ms=latency_ms,
            policy_version=self.policy.policy_version,
            metadata={}
        )
        self.telemetry_sink.record(result, latency_ms)
        return result
