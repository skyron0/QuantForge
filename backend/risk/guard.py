import uuid
import time
from datetime import datetime, timezone
from typing import Tuple, List, Optional
import math

from backend.decision.models import TradeProposal
from backend.risk.models import RiskContext, RiskAuthorizationResult, RiskAuthorizationStatus
from backend.risk.policy import RiskPolicy
from backend.risk.exceptions import (
    RiskError,
    RiskValidationError,
    ProposalExpiredError,
    RiskContextError,
    RiskAuthorizationError,
)
from backend.risk.telemetry import RiskTelemetrySink


class RiskGuardEngine:

    def __init__(self, policy: RiskPolicy, telemetry_sink: Optional[RiskTelemetrySink] = None):
        self.policy = policy
        self.telemetry_sink = telemetry_sink

    def evaluate(
        self,
        proposal: TradeProposal,
        context: RiskContext,
        current_time: Optional[datetime] = None,
    ) -> RiskAuthorizationResult:
        t_start = time.perf_counter()

        # Audit lineage setup
        auth_id = str(uuid.uuid4())
        now = current_time or datetime.now(timezone.utc)
        evaluated_at = now.isoformat()

        rejection_reasons: List[str] = []
        adjustment_reasons: List[str] = []
        triggered_rules: List[str] = []

        # 1. Input integrity & structural checks
        try:
            self._validate_inputs(proposal, context)
        except RiskError as e:
            # Immediate fail closed (critical input corruption)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            error_result = RiskAuthorizationResult(
                authorization_id=auth_id,
                proposal_id=getattr(proposal, "proposal_id", "unknown"),
                symbol=getattr(proposal, "symbol", "unknown"),
                direction=getattr(proposal, "direction", "UNKNOWN"),
                status=RiskAuthorizationStatus.REJECTED,
                original_confidence=getattr(proposal, "confidence", 0.0),
                effective_confidence=0.0,
                rejection_reasons=[str(e)],
                adjustment_reasons=[],
                triggered_rules=["INPUT_INTEGRITY_CHECK"],
                policy_version=self.policy.policy_version,
                source_model_version=getattr(proposal, "source_model_version", "unknown"),
                fusion_policy_version=getattr(proposal, "fusion_policy_version", "unknown"),
                proposal_created_at=getattr(proposal, "created_at", evaluated_at),
                evaluated_at=evaluated_at,
                latency_ms=latency_ms,
                requested_risk_fraction=self.policy.base_risk_fraction,
                authorized_risk_fraction=0.0,
                reasoning_request_id=getattr(proposal, "reasoning_request_id", None),
            )
            if self.telemetry_sink:
                self.telemetry_sink.record(error_result, context, latency_ms)
            return error_result

        # 2. Proposal freshness gate
        try:
            proposal_dt = datetime.fromisoformat(proposal.created_at)
        except ValueError:
            rejection_reasons.append("Invalid proposal created_at timestamp format")
            triggered_rules.append("PROPOSAL_FRESHNESS_GATE")
            proposal_dt = now

        if (proposal_dt - now).total_seconds() > 5.0:
            rejection_reasons.append(
                f"Proposal is in the future relative to evaluation: "
                f"proposal={proposal.created_at}, now={evaluated_at}"
            )
            triggered_rules.append("PROPOSAL_FRESHNESS_GATE")

        age_seconds = (now - proposal_dt).total_seconds()
        if age_seconds > self.policy.maximum_proposal_age_seconds:
            rejection_reasons.append(
                f"Proposal is too stale: age={age_seconds:.2f}s, "
                f"limit={self.policy.maximum_proposal_age_seconds:.2f}s"
            )
            triggered_rules.append("PROPOSAL_FRESHNESS_GATE")

        try:
            expires_dt = datetime.fromisoformat(proposal.expires_at)
            if now >= expires_dt:
                rejection_reasons.append(
                    f"Proposal has expired: expires_at={proposal.expires_at}, "
                    f"now={evaluated_at}"
                )
                triggered_rules.append("PROPOSAL_FRESHNESS_GATE")
        except ValueError:
            rejection_reasons.append("Invalid proposal expires_at timestamp format")
            triggered_rules.append("PROPOSAL_FRESHNESS_GATE")

        # 3. Confidence validation
        if proposal.confidence < self.policy.minimum_proposal_confidence:
            rejection_reasons.append(
                f"Proposal confidence ({proposal.confidence:.2f}) under minimum "
                f"({self.policy.minimum_proposal_confidence:.2f})"
            )
            triggered_rules.append("CONFIDENCE_GATE")

        # 4. Critical ML Drift Gate (Defense-in-depth)
        drift = proposal.metadata.get("drift_status", "normal")
        if drift == "critical" and self.policy.reject_on_critical_drift:
            rejection_reasons.append("Critical ML model drift detected in proposal lineage")
            triggered_rules.append("ML_DRIFT_GATE")

        # 5. Daily Loss Gate
        # PNL fraction calculation: realized + unrealized relative to equity
        total_daily_pnl = context.daily_realized_pnl + context.daily_unrealized_pnl
        pnl_fraction = total_daily_pnl / context.equity
        # Reject if daily loss is worse than or equal to negative limit
        if pnl_fraction <= -self.policy.maximum_daily_loss_fraction:
            rejection_reasons.append(
                f"Daily loss limit reached/exceeded: current={pnl_fraction:.4f}, "
                f"limit={-self.policy.maximum_daily_loss_fraction:.4f}"
            )
            triggered_rules.append("DAILY_LOSS_GATE")

        # 6. Drawdown Gate
        if context.current_drawdown_pct >= self.policy.maximum_drawdown_fraction:
            rejection_reasons.append(
                f"Drawdown limit reached/exceeded: current={context.current_drawdown_pct:.4f}, "
                f"limit={self.policy.maximum_drawdown_fraction:.4f}"
            )
            triggered_rules.append("DRAWDOWN_GATE")

        # 7. Portfolio Exposure Gate
        if context.portfolio_exposure_pct >= self.policy.maximum_portfolio_exposure_fraction:
            rejection_reasons.append(
                f"Portfolio exposure limit violated: current={context.portfolio_exposure_pct:.4f}, "
                f"limit={self.policy.maximum_portfolio_exposure_fraction:.4f}"
            )
            triggered_rules.append("PORTFOLIO_EXPOSURE_GATE")

        # 8. Symbol Exposure Gate
        if context.symbol_exposure_pct >= self.policy.maximum_symbol_exposure_fraction:
            rejection_reasons.append(
                f"Symbol exposure limit violated: current={context.symbol_exposure_pct:.4f}, "
                f"limit={self.policy.maximum_symbol_exposure_fraction:.4f}"
            )
            triggered_rules.append("SYMBOL_EXPOSURE_GATE")

        # 9. Leverage Gate
        if context.current_leverage >= self.policy.maximum_leverage:
            rejection_reasons.append(
                f"Leverage limit reached: current={context.current_leverage:.2f}, "
                f"limit={self.policy.maximum_leverage:.2f}"
            )
            triggered_rules.append("LEVERAGE_GATE")

        # 10. Open Position limits
        if context.open_positions_count >= self.policy.maximum_open_positions:
            rejection_reasons.append(
                f"Maximum open positions limit reached: current={context.open_positions_count}, "
                f"limit={self.policy.maximum_open_positions}"
            )
            triggered_rules.append("OPEN_POSITIONS_GATE")

        # 11. Symbol Position limits
        if context.symbol_open_positions_count >= self.policy.maximum_symbol_open_positions:
            rejection_reasons.append(
                f"Maximum open positions for symbol reached: "
                f"current={context.symbol_open_positions_count}, "
                f"limit={self.policy.maximum_symbol_open_positions}"
            )
            triggered_rules.append("SYMBOL_POSITIONS_GATE")

        # 12. Consecutive losses safety (circuit breaker)
        if context.consecutive_losses >= self.policy.maximum_consecutive_losses:
            rejection_reasons.append(
                f"Consecutive losses circuit breaker active: current={context.consecutive_losses}, "
                f"limit={self.policy.maximum_consecutive_losses}"
            )
            triggered_rules.append("CONSECUTIVE_LOSS_GATE")

        # 13. Volatility safety gate
        if context.volatility_state == "CRITICAL" and self.policy.reject_on_critical_volatility:
            rejection_reasons.append("Critical market volatility detected")
            triggered_rules.append("VOLATILITY_SAFETY_GATE")

        # 14. Liquidity safety gate
        if context.market_liquidity_state == "CRITICAL" and self.policy.reject_on_critical_liquidity:
            rejection_reasons.append("Critical market liquidity detected")
            triggered_rules.append("LIQUIDITY_SAFETY_GATE")

        # 15. Risk Flag validations
        proposal_flags = set(proposal.risk_flags)

        # Checking Blocking Flags
        blocking_triggered = proposal_flags.intersection(self.policy.blocking_risk_flags)
        if blocking_triggered:
            rejection_reasons.append(
                f"Blocking risk flags detected: {sorted(list(blocking_triggered))}"
            )
            triggered_rules.append("BLOCKING_RISK_FLAGS")

        # Unrecognized logic (fail-closed for potential safety-sensitive flags)
        all_policy_flags = (
            self.policy.blocking_risk_flags
            .union(self.policy.risk_reducing_flags.keys())
            .union(self.policy.informational_risk_flags)
        )
        unrecognized_flags = proposal_flags.difference(all_policy_flags)
        if unrecognized_flags:
            rejection_reasons.append(
                f"Unrecognized safety-sensitive flags: {sorted(list(unrecognized_flags))}"
            )
            triggered_rules.append("UNRECOGNIZED_RISK_FLAGS")

        # 16. Decision / Adjustments (if no rejections)
        status = RiskAuthorizationStatus.APPROVED
        effective_confidence = proposal.confidence
        authorized_risk_fraction = self.policy.base_risk_fraction

        if not rejection_reasons:
            # Volatility multiplier adjustment
            vol_mult = self.policy.volatility_adjustments.get(context.volatility_state, 1.0)
            if vol_mult < 1.0:
                authorized_risk_fraction *= vol_mult
                adjustment_reasons.append(
                    f"Volatility adjustment for state {context.volatility_state} (* {vol_mult:.2f})"
                )
                triggered_rules.append("VOLATILITY_ADJUSTMENT")
                status = RiskAuthorizationStatus.ADJUSTED

            # Risk reducing flags multiplier adjustment
            for flag in proposal.risk_flags:
                flag_mult = self.policy.risk_reducing_flags.get(flag, 1.0)
                if flag_mult < 1.0:
                    authorized_risk_fraction *= flag_mult
                    adjustment_reasons.append(
                        f"Risk reducing flag multiplier for '{flag}' (* {flag_mult:.2f})"
                    )
                    triggered_rules.append("RISK_REDUCING_FLAG_ADJUSTMENT")
                    status = RiskAuthorizationStatus.ADJUSTED

            # Enforce max cap
            if authorized_risk_fraction > self.policy.maximum_risk_fraction:
                authorized_risk_fraction = self.policy.maximum_risk_fraction
                adjustment_reasons.append(
                    f"Capped to policy maximum_risk_fraction ({self.policy.maximum_risk_fraction:.4f})"
                )
                triggered_rules.append("MAX_RISK_CAP")
                status = RiskAuthorizationStatus.ADJUSTED

            # Enforce min floor (reject if below)
            if authorized_risk_fraction < self.policy.minimum_risk_fraction:
                rejection_reasons.append(
                    f"Adjusted risk fraction ({authorized_risk_fraction:.6f}) fell below "
                    f"minimum_risk_fraction ({self.policy.minimum_risk_fraction:.6f})"
                )
                triggered_rules.append("MIN_RISK_FLOOR_REJECT")
                status = RiskAuthorizationStatus.REJECTED
                authorized_risk_fraction = 0.0
                effective_confidence = 0.0

        if rejection_reasons:
            status = RiskAuthorizationStatus.REJECTED
            authorized_risk_fraction = 0.0
            effective_confidence = 0.0

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        result = RiskAuthorizationResult(
            authorization_id=auth_id,
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            direction=proposal.direction,
            status=status,
            original_confidence=proposal.confidence,
            effective_confidence=effective_confidence,
            rejection_reasons=rejection_reasons,
            adjustment_reasons=adjustment_reasons,
            triggered_rules=triggered_rules,
            policy_version=self.policy.policy_version,
            source_model_version=proposal.source_model_version,
            fusion_policy_version=proposal.fusion_policy_version,
            proposal_created_at=proposal.created_at,
            evaluated_at=evaluated_at,
            latency_ms=latency_ms,
            requested_risk_fraction=self.policy.base_risk_fraction,
            authorized_risk_fraction=authorized_risk_fraction,
            reasoning_request_id=proposal.reasoning_request_id,
            metadata=proposal.metadata or {},
        )

        if self.telemetry_sink:
            self.telemetry_sink.record(result, context, latency_ms)

        return result

    def _validate_inputs(self, proposal: TradeProposal, context: RiskContext) -> None:
        if not isinstance(proposal, TradeProposal):
            raise RiskContextError("Input proposal is not an instance of TradeProposal")
        if not isinstance(context, RiskContext):
            raise RiskContextError("Input context is not an instance of RiskContext")

        # Checks basic bounds on TradeProposal fields
        if not (0.0 <= proposal.confidence <= 1.0):
            raise RiskContextError(
                f"Proposal confidence ({proposal.confidence}) must be in range [0.0, 1.0]"
            )
        if proposal.direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
            raise RiskContextError(f"Unknown direction in proposal: {proposal.direction}")
        if not proposal.symbol:
            raise RiskContextError("Proposal must specify a symbol")
        if not proposal.proposal_id:
            raise RiskContextError("Proposal is missing proposal_id")
        if proposal.symbol != context.symbol:
            raise RiskContextError(
                f"Symbol mismatch: proposal={proposal.symbol}, context={context.symbol}"
            )
