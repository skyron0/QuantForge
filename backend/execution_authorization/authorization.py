import time
import math
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable

from backend.replay.clock import Clock, SystemClock

# Decoupled domain inputs
from backend.decision.models import TradeProposal
from backend.risk.models import RiskAuthorizationResult, RiskAuthorizationStatus
from backend.positioning.models import PositionSizeResult

# Local package components
from backend.execution_authorization.exceptions import (
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
    LiveExecutionNotAllowedError
)
from backend.execution_authorization.models import (
    ExecutionEnvironment,
    OrderDirection,
    OrderType,
    ExecutionAuthorizationStatus,
    ExecutionContext,
    OrderIntent,
    ExecutionAuthorizationResult as EngineResult
)
from backend.execution_authorization.policy import ExecutionPolicy
from backend.execution_authorization.idempotency import IdempotencyStore
from backend.execution_authorization.telemetry import ExecutionAuthorizationTelemetrySink


def parse_iso(ts: str) -> datetime:
    """Helper to parse ISO strings, replacing Z with UTC timezone offset."""
    if not ts:
        raise ExecutionValidationError("Empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError as e:
        raise ExecutionValidationError(f"Invalid timestamp format: {ts}") from e


class ExecutionAuthorizationEngine:
    """
    Deterministic execution authorization gate.
    The final fence before exchange adapters.
    """

    def __init__(
        self,
        policy: ExecutionPolicy,
        idempotency_store: IdempotencyStore,
        telemetry_sink: Optional[ExecutionAuthorizationTelemetrySink] = None,
        clock: Optional[Clock] = None,
        uuid_generator: Optional[Callable[[], str]] = None
    ):
        if not isinstance(policy, ExecutionPolicy):
            raise InvalidExecutionPolicyError("Invalid Policy provided")
        self.policy = policy
        self.idempotency_store = idempotency_store
        self.telemetry_sink = telemetry_sink
        self.clock = clock or SystemClock()
        self._uuid_generator = uuid_generator or (lambda: str(uuid.uuid4()))

    def evaluate(
        self,
        proposal: TradeProposal,
        risk_auth: RiskAuthorizationResult,
        size_res: PositionSizeResult,
        context: ExecutionContext
    ) -> EngineResult:
        start_counter = time.perf_counter()
        triggered_rules: List[str] = []

        try:
            # 1. Parameter Checks (Fail-Closed on None)
            if proposal is None or risk_auth is None or size_res is None or context is None:
                raise ExecutionValidationError("All parameters must be provided")

            # Validate context values
            if context.available_balance is None or context.current_price is None:
                raise ExecutionValidationError("Available balance and current price must not be None")

            # Parse timestamps
            curr_dt = parse_iso(context.current_timestamp)
            mkt_dt = parse_iso(context.market_timestamp)

            # 2. Complete Lineage Validation
            if risk_auth.proposal_id != proposal.proposal_id:
                raise LineageMismatchError(
                    f"Risk proposal ID {risk_auth.proposal_id} mismatch with Proposal ID {proposal.proposal_id}"
                )
            if size_res.proposal_id != proposal.proposal_id:
                raise LineageMismatchError(
                    f"Sizing proposal ID {size_res.proposal_id} mismatch with Proposal ID {proposal.proposal_id}"
                )
            if size_res.authorization_id != risk_auth.authorization_id:
                raise LineageMismatchError(
                    f"Sizing auth ID {size_res.authorization_id} mismatch with Risk Auth ID {risk_auth.authorization_id}"
                )

            # Symbol match
            if not (proposal.symbol == risk_auth.symbol == size_res.symbol):
                raise LineageMismatchError(
                    f"Symbol mismatch across components: Proposal={proposal.symbol}, "
                    f"Risk={risk_auth.symbol}, Sizing={size_res.symbol}"
                )

            # Direction match
            if not (proposal.direction == risk_auth.direction == size_res.direction):
                raise LineageMismatchError(
                    f"Direction mismatch across components: Proposal={proposal.direction}, "
                    f"Risk={risk_auth.direction}, Sizing={size_res.direction}"
                )

            # 2.5 TradeProposal expiry check
            proposal_expiry_dt = parse_iso(proposal.expires_at)
            if curr_dt > proposal_expiry_dt:
                triggered_rules.append("PROPOSAL_EXPIRED")
                raise ExecutionValidationError(
                    f"Trade proposal is expired: current {curr_dt} > expiry {proposal_expiry_dt}"
                )

            # 3. Direct Direction Mapping
            # Decision/Risk uses BULLISH / BEARISH / NEUTRAL
            prop_dir = proposal.direction.upper()
            if prop_dir == "NEUTRAL":
                raise ExecutionValidationError("NEUTRAL proposals are rejected")
            elif prop_dir == "BULLISH":
                mapped_direction = OrderDirection.BUY
            elif prop_dir == "BEARISH":
                mapped_direction = OrderDirection.SELL
            else:
                raise ExecutionValidationError(f"Invalid direction state: {proposal.direction}")

            # 4. Risk Guard Rejection Check
            if risk_auth.status == RiskAuthorizationStatus.REJECTED:
                raise ExecutionValidationError("Associated risk authorization was REJECTED")

            # Ensure status is approved or adjusted
            if risk_auth.status not in (RiskAuthorizationStatus.APPROVED, RiskAuthorizationStatus.ADJUSTED):
                raise ExecutionValidationError(f"Invalid risk authorization status: {risk_auth.status}")

            # 5. Position Sizing Validation
            if size_res.quantity <= 0 or not math.isfinite(size_res.quantity):
                raise ExecutionValidationError("Sizing quantity must be finite and greater than zero")

            # Quantity bounds check
            if size_res.quantity < self.policy.minimum_quantity:
                raise ExecutionValidationError(
                    f"Quantity {size_res.quantity} is below policy minimum {self.policy.minimum_quantity}"
                )
            if size_res.quantity > self.policy.maximum_quantity:
                raise ExecutionValidationError(
                    f"Quantity {size_res.quantity} exceeds policy maximum {self.policy.maximum_quantity}"
                )

            # Check that actual risk fraction does not exceed authorization
            # Allowing for minor floating point error, but strict check otherwise
            if size_res.authorized_risk_fraction > risk_auth.authorized_risk_fraction + 1e-9:
                raise ExecutionValidationError(
                    f"Sizing authorized risk fraction {size_res.authorized_risk_fraction} "
                    f"exceeds authorized RiskGuard fraction {risk_auth.authorized_risk_fraction}"
                )

            # 6. Global Enablement & Kill-Switch Validation
            if self.policy.require_execution_enabled and not context.execution_enabled:
                triggered_rules.append("EXECUTION_DISABLED")
                raise ExecutionDisabledError("Global execution is disabled")

            if self.policy.require_symbol_enabled and not context.symbol_trading_enabled:
                triggered_rules.append("SYMBOL_DISABLED")
                raise SymbolTradingDisabledError(f"Symbol trading is disabled for {proposal.symbol}")

            if self.policy.reject_when_kill_switch_active and context.kill_switch_active:
                triggered_rules.append("KILL_SWITCH")
                raise KillSwitchActiveError("Kill switch is active")

            # 7. Environment Safety Opt-in
            if context.environment not in self.policy.allowed_environments:
                triggered_rules.append("ENVIRONMENT_NOT_ALLOWED")
                raise LiveExecutionNotAllowedError(
                    f"Environment {context.environment} is not allowed by execution policy"
                )

            if context.environment == ExecutionEnvironment.LIVE:
                if not self.policy.allow_live_execution_intents:
                    triggered_rules.append("LIVE_NOT_PERMITTED")
                    raise LiveExecutionNotAllowedError(
                        "Live execution intent is rejected by default. Explicit policy configuration required."
                    )

            # 8. Market Data Freshness
            # Clock skew check
            if mkt_dt > curr_dt:
                skew = (mkt_dt - curr_dt).total_seconds()
                if skew > self.policy.maximum_clock_skew_seconds:
                    triggered_rules.append("CLOCK_SKEW_LIMIT")
                    raise StaleMarketDataError(
                        f"Market timestamp is in the future. Skew of {skew}s exceeds max allowance of {self.policy.maximum_clock_skew_seconds}s"
                    )

            # Stale market data check
            age = (curr_dt - mkt_dt).total_seconds()
            if age > self.policy.maximum_market_data_age_seconds:
                triggered_rules.append("STALE_MARKET_DATA")
                raise StaleMarketDataError(
                    f"Stale market data: age of {age}s exceeds policy maximum of {self.policy.maximum_market_data_age_seconds}s"
                )

            # 9. Stop Loss / Take Profit Validations
            bp_sl = size_res.stop_loss_price
            bp_tp = proposal.metadata.get("take_profit", None)

            # Stop loss configuration validation
            if self.policy.require_stop_loss:
                if bp_sl is None:
                    raise ExecutionValidationError("Stop loss price is required by policy but missing")

            if bp_sl is not None:
                if math.isnan(bp_sl) or math.isinf(bp_sl) or bp_sl <= 0:
                    raise ExecutionValidationError(f"Invalid stop loss price: {bp_sl}")
                # Direction orientation checks
                if mapped_direction == OrderDirection.BUY and bp_sl >= size_res.entry_price:
                    raise ExecutionValidationError(
                        f"BUY Stop loss {bp_sl} must be below entry price {size_res.entry_price}"
                    )
                if mapped_direction == OrderDirection.SELL and bp_sl <= size_res.entry_price:
                    raise ExecutionValidationError(
                        f"SELL Stop loss {bp_sl} must be above entry price {size_res.entry_price}"
                    )

            # Take profit configuration validation
            if self.policy.require_take_profit:
                if bp_tp is None:
                    raise ExecutionValidationError("Take profit price is required by policy but missing")

            if bp_tp is not None:
                if math.isnan(bp_tp) or math.isinf(bp_tp) or bp_tp <= 0:
                    raise ExecutionValidationError(f"Invalid take profit price: {bp_tp}")
                # Direction orientation checks
                if mapped_direction == OrderDirection.BUY and bp_tp <= size_res.entry_price:
                    raise ExecutionValidationError(
                        f"BUY Take profit {bp_tp} must be above entry price {size_res.entry_price}"
                    )
                if mapped_direction == OrderDirection.SELL and bp_tp >= size_res.entry_price:
                    raise ExecutionValidationError(
                        f"SELL Take profit {bp_tp} must be below entry price {size_res.entry_price}"
                    )

            # 10. Order Type Validations
            # Limit price checking dependent on type
            # Find proposal order type from proposal metadata or default to MARKET.
            ot_str = proposal.metadata.get("order_type", "MARKET").upper()
            if ot_str not in [ot.value for ot in self.policy.allowed_order_types]:
                raise ExecutionValidationError(f"Order type {ot_str} is not allowed by policy")

            order_type = OrderType(ot_str)
            limit_price = proposal.metadata.get("limit_price", None)

            if order_type == OrderType.MARKET:
                if limit_price is not None:
                    raise ExecutionValidationError("MARKET order cannot contain a limit price")
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ExecutionValidationError("LIMIT order requires a limit price")
                if not isinstance(limit_price, (int, float)):
                    raise ExecutionValidationError(f"Limit price must be a number, got {type(limit_price)}")
                if math.isnan(limit_price) or math.isinf(limit_price) or limit_price <= 0:
                    raise ExecutionValidationError(f"Invalid limit price: {limit_price}")

            # 11. Idempotency Check (Passed Gates, now atomic reserve)
            # Keys based on: proposal + risk_auth + sizing + environment
            idempotency_key = f"{proposal.proposal_id}:{risk_auth.authorization_id}:{size_res.sizing_id}:{context.environment.value}"

            # Atomically register key. TTL based on policy's order_intent_ttl_seconds
            success = self.idempotency_store.register_if_absent(
                idempotency_key=idempotency_key,
                data=True,
                ttl_seconds=self.policy.order_intent_ttl_seconds
            )
            if not success:
                triggered_rules.append("DUPLICATE_INTENT")
                raise DuplicateIntentError(
                    f"Duplicate execution request detected for idempotency key {idempotency_key}"
                )

            # 12. Create OrderIntent
            try:
                intent_id = self._uuid_generator()
                created_at = context.current_timestamp
                # Calculate expires_at
                created_dt = parse_iso(created_at)
                expires_dt = datetime.fromtimestamp(
                    created_dt.timestamp() + self.policy.order_intent_ttl_seconds,
                    tz=timezone.utc
                )
                expires_at = expires_dt.isoformat().replace("+00:00", "Z")

                intent = OrderIntent(
                    intent_id=intent_id,
                    idempotency_key=idempotency_key,
                    proposal_id=proposal.proposal_id,
                    risk_authorization_id=risk_auth.authorization_id,
                    sizing_id=size_res.sizing_id,
                    symbol=proposal.symbol,
                    direction=mapped_direction,
                    quantity=size_res.quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_loss=bp_sl,
                    take_profit=bp_tp,
                    environment=context.environment,
                    source_model_version=proposal.source_model_version,
                    fusion_policy_version=proposal.fusion_policy_version,
                    risk_policy_version=risk_auth.policy_version,
                    position_sizing_policy_version=size_res.policy_version,
                    execution_policy_version=self.policy.policy_version,
                    reasoning_request_id=proposal.reasoning_request_id,
                    created_at=created_at,
                    expires_at=expires_at,
                    metadata=proposal.metadata or {}
                )
            except Exception as construction_error:
                # 13. Idempotency Rollback Behavior
                self.idempotency_store.invalidate(idempotency_key)
                raise construction_error

            latency_ms = (time.perf_counter() - start_counter) * 1000.0

            result = EngineResult(
                authorization_id=self._uuid_generator(),
                status=ExecutionAuthorizationStatus.AUTHORIZED,
                intent=intent,
                rejection_reason="",
                triggered_rules=[],
                policy_version=self.policy.policy_version,
                proposal_id=proposal.proposal_id,
                risk_authorization_id=risk_auth.authorization_id,
                sizing_id=size_res.sizing_id,
                latency_ms=latency_ms,
                timestamp=self.clock.now().isoformat().replace("+00:00", "Z"),
                metadata={}
            )

            if self.telemetry_sink:
                self._record_telemetry(result, idempotency_key, mapped_direction, size_res.quantity, proposal.symbol, context.environment, age)

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            rejection_reason = str(e)
            # Determine triggered rule if not set
            if not triggered_rules:
                triggered_rules.append(type(e).__name__)

            result = EngineResult(
                authorization_id=self._uuid_generator(),
                status=ExecutionAuthorizationStatus.REJECTED,
                intent=None,
                rejection_reason=rejection_reason,
                triggered_rules=triggered_rules,
                policy_version=self.policy.policy_version,
                proposal_id=proposal.proposal_id if proposal else "UNKNOWN",
                risk_authorization_id=risk_auth.authorization_id if risk_auth else "UNKNOWN",
                sizing_id=size_res.sizing_id if size_res else "UNKNOWN",
                latency_ms=latency_ms,
                timestamp=self.clock.now().isoformat().replace("+00:00", "Z"),
                metadata={}
            )

            # Clean/safe telemetry logging
            if self.telemetry_sink:
                self._record_telemetry_rejection(result, proposal, risk_auth, size_res, context, latency_ms, rejection_reason, triggered_rules)

            # We must fail closed by returning the rejected authorization result
            return result

    def _record_telemetry(self, res: EngineResult, idempotency_key: str, direction: OrderDirection, qty: float, symbol: str, environment: ExecutionEnvironment, market_data_age: float) -> None:
        if self.telemetry_sink:
            self.telemetry_sink.record_authorization({
                "authorization_id": res.authorization_id,
                "intent_id": res.intent.intent_id if res.intent else None,
                "idempotency_key": idempotency_key,
                "proposal_id": res.proposal_id,
                "risk_authorization_id": res.risk_authorization_id,
                "sizing_id": res.sizing_id,
                "environment": environment.value,
                "symbol": symbol,
                "direction": direction.value,
                "quantity": qty,
                "policy_version": res.policy_version,
                "status": res.status.value,
                "triggered_rules": res.triggered_rules,
                "rejection_reason": res.rejection_reason,
                "market_data_age": market_data_age,
                "latency_ms": res.latency_ms,
                "timestamp": res.timestamp
            })

    def _record_telemetry_rejection(
        self,
        res: EngineResult,
        proposal: Optional[TradeProposal],
        risk_auth: Optional[RiskAuthorizationResult],
        size_res: Optional[PositionSizeResult],
        context: Optional[ExecutionContext],
        latency_ms: float,
        rejection_reason: str,
        triggered_rules: List[str]
    ) -> None:
        # Avoid logging credentials or passwords
        if self.telemetry_sink:
            self.telemetry_sink.record_authorization({
                "authorization_id": res.authorization_id,
                "intent_id": None,
                "idempotency_key": f"{proposal.proposal_id if proposal else 'UNKNOWN'}:{risk_auth.authorization_id if risk_auth else 'UNKNOWN'}:{size_res.sizing_id if size_res else 'UNKNOWN'}:{context.environment.value if context else 'UNKNOWN'}",
                "proposal_id": res.proposal_id,
                "risk_authorization_id": res.risk_authorization_id,
                "sizing_id": res.sizing_id,
                "environment": context.environment.value if context else "UNKNOWN",
                "symbol": proposal.symbol if proposal else "UNKNOWN",
                "direction": proposal.direction if proposal else "UNKNOWN",
                "quantity": size_res.quantity if size_res else 0.0,
                "policy_version": res.policy_version,
                "status": res.status.value,
                "triggered_rules": triggered_rules,
                "rejection_reason": rejection_reason,
                "market_data_age": -1.0,
                "latency_ms": latency_ms,
                "timestamp": res.timestamp
            })
