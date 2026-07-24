import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any, Callable

from backend.replay.clock import Clock, SystemClock

# Sprint 3.3 execution authorization imports
from backend.execution_authorization.models import (
    ExecutionContext,
    OrderIntent,
    OrderType,
    ExecutionAuthorizationStatus,
    ExecutionAuthorizationResult as EngineResult
)
from backend.execution_authorization.policy import ExecutionPolicy
from backend.execution_authorization.exceptions import (
    ExecutionValidationError,
    StaleMarketDataError,
    LineageMismatchError,
    ExecutionDisabledError,
    KillSwitchActiveError,
    SymbolTradingDisabledError,
    LiveExecutionNotAllowedError
)
# Position lifecycle imports
from backend.position_lifecycle.models import ExitProposal


def parse_iso(ts: str) -> datetime:
    if not ts:
        raise ValueError("Empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {ts}") from e


class ExitExecutionRequestBuilder:
    """
    Decoupled builder that maps an authorized ExitProposal to an OrderIntent.
    No execution logic is defined here.
    """
    @staticmethod
    def build_intent(
        proposal: ExitProposal,
        execution_policy: ExecutionPolicy,
        risk_policy_version: str,
        sizing_policy_version: str,
        idempotency_key: str,
        uuid_generator: Optional[Callable[[], str]] = None
    ) -> OrderIntent:
        # Strict validation
        if not proposal or not execution_policy:
            raise ValueError("Proposal and execution policy are required")
            
        _uuid_gen = uuid_generator or (lambda: str(uuid.uuid4()))
            
        return OrderIntent(
            intent_id=_uuid_gen(),
            idempotency_key=idempotency_key,
            proposal_id=proposal.exit_proposal_id,
            risk_authorization_id="RISK_EXIT_AUTH_" + _uuid_gen()[:8],
            sizing_id="SIZE_EXIT_" + _uuid_gen()[:8],
            symbol=proposal.symbol,
            direction=proposal.exit_direction,
            quantity=float(proposal.requested_quantity),
            order_type=OrderType.MARKET,  # Protective exits are usually MARKET orders
            limit_price=None,
            stop_loss=None,
            take_profit=None,
            environment=context_env_map(execution_policy),
            source_model_version="exit_engine",
            fusion_policy_version="exit_engine",
            risk_policy_version=risk_policy_version,
            position_sizing_policy_version=sizing_policy_version,
            execution_policy_version=execution_policy.policy_version,
            reasoning_request_id=None,
            created_at=proposal.created_at,
            expires_at=proposal.expires_at,
            metadata=proposal.metadata or {}
        )


def context_env_map(exec_policy: ExecutionPolicy):
    # Mapping environment helper to get ExecutionEnvironment Enum matching policy.allowed_environments
    from backend.execution_authorization.models import ExecutionEnvironment
    for env_name in ["PAPER", "SHADOW", "LIVE"]:
        if env_name in exec_policy.allowed_environments:
            return ExecutionEnvironment(env_name)
    return ExecutionEnvironment.PAPER


class ExitAuthorizationEngine:
    """
    Validation and authorization boundary verifying that ExitProposals
    comply with Sprint 3.3 safety guards before converting to OrderIntents.
    """
    def __init__(
        self,
        execution_policy: ExecutionPolicy,
        clock: Optional[Clock] = None,
        uuid_generator: Optional[Callable[[], str]] = None
    ):
        self.execution_policy = execution_policy
        self.clock = clock or SystemClock()
        self._uuid_generator = uuid_generator or (lambda: str(uuid.uuid4()))

    def authorize_exit(
        self,
        proposal: ExitProposal,
        context: ExecutionContext,
        risk_policy_version: str,
        sizing_policy_version: str,
        idempotency_key: str
    ) -> EngineResult:
        start_counter = time.perf_counter()
        triggered_rules: List[str] = []

        try:
            # 1. Parameter validation
            if not proposal or not context:
                raise ExecutionValidationError("Proposal and Context must be supplied")

            # 2. Match symbol/lineage
            if proposal.symbol != context.metadata.get("symbol", proposal.symbol):
                raise LineageMismatchError(
                    f"Symbol mismatch: proposal={proposal.symbol}, context={context.metadata.get('symbol')}"
                )

            # 3. Global execution enablement gate
            if not context.execution_enabled:
                raise ExecutionDisabledError("Global execution is disabled")
            if context.kill_switch_active:
                raise KillSwitchActiveError("Kill switch is active")
            if not context.symbol_trading_enabled:
                raise SymbolTradingDisabledError(f"Trading disabled for symbol: {proposal.symbol}")

            # 4. Environment constraints
            # If the context demands LIVE but policy does not authorize it
            from backend.execution_authorization.models import ExecutionEnvironment
            if context.environment == ExecutionEnvironment.LIVE:
                if "LIVE" not in self.execution_policy.allowed_environments:
                    raise LiveExecutionNotAllowedError("LIVE execution environment not allowed by policy")

            # 5. Expiry Check
            prop_exp_dt = parse_iso(proposal.expires_at)
            sys_dt = parse_iso(context.current_timestamp)
            if sys_dt > prop_exp_dt:
                raise ExecutionValidationError("Exit proposal has expired")

            # 6. Quality Checks
            if float(proposal.requested_quantity) < self.execution_policy.minimum_quantity:
                raise ExecutionValidationError(
                    f"Exit quantity {proposal.requested_quantity} is below policy minimum {self.execution_policy.minimum_quantity}"
                )

            # Build Authorized Intent
            intent = ExitExecutionRequestBuilder.build_intent(
                proposal=proposal,
                execution_policy=self.execution_policy,
                risk_policy_version=risk_policy_version,
                sizing_policy_version=sizing_policy_version,
                idempotency_key=idempotency_key,
                uuid_generator=self._uuid_generator
            )

            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return EngineResult(
                authorization_id="EXIT_AUTH_" + self._uuid_generator(),
                status=ExecutionAuthorizationStatus.AUTHORIZED,
                intent=intent,
                rejection_reason="",
                triggered_rules=triggered_rules,
                policy_version=self.execution_policy.policy_version,
                proposal_id=proposal.exit_proposal_id,
                risk_authorization_id=intent.risk_authorization_id,
                sizing_id=intent.sizing_id,
                latency_ms=latency_ms,
                timestamp=context.current_timestamp,
                metadata={"trigger_type": proposal.trigger_type.value}
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return EngineResult(
                authorization_id="EXIT_REJ_" + self._uuid_generator(),
                status=ExecutionAuthorizationStatus.REJECTED,
                intent=None,
                rejection_reason=str(e),
                triggered_rules=triggered_rules,
                policy_version=self.execution_policy.policy_version,
                proposal_id=proposal.exit_proposal_id if proposal else "N/A",
                risk_authorization_id="N/A",
                sizing_id="N/A",
                latency_ms=latency_ms,
                timestamp=context.current_timestamp if context else self.clock.now().isoformat()
            )
