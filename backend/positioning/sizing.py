import uuid
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Optional

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


class PositionSizingEngine:
    """Deterministic, risk-based position sizing engine."""

    def __init__(
        self,
        policy: PositionSizingPolicy,
        telemetry_sink: Optional[PositionSizingTelemetrySink] = None,
    ):
        self.policy = policy
        self.telemetry_sink = telemetry_sink

    def evaluate(
        self,
        proposal: TradeProposal,
        authorization: RiskAuthorizationResult,
        context: PositionSizingContext,
    ) -> PositionSizeResult:
        start_time = time.perf_counter()
        rejection_reason = ""
        success = False
        result: Optional[PositionSizeResult] = None

        try:
            # 1. Authorization Verification
            if not authorization or not proposal or not context:
                raise AuthorizationError("Missing required input parameters")

            # Check status
            if authorization.status not in (RiskAuthorizationStatus.APPROVED, RiskAuthorizationStatus.ADJUSTED):
                raise AuthorizationError(f"Proposal authorization status is invalid: {authorization.status}")

            # Match proposal
            if authorization.proposal_id != proposal.proposal_id:
                raise AuthorizationError(
                    f"Mismatched proposal_id: auth={authorization.proposal_id}, proposal={proposal.proposal_id}"
                )

            # Match symbol
            if authorization.symbol != proposal.symbol or authorization.symbol != context.symbol:
                raise AuthorizationError(
                    f"Mismatched symbols: auth={authorization.symbol}, proposal={proposal.symbol}, context={context.symbol}"
                )

            # Validate risk fraction
            auth_risk = authorization.authorized_risk_fraction
            if math.isnan(auth_risk) or math.isinf(auth_risk) or auth_risk <= 0.0 or auth_risk > 1.0:
                raise AuthorizationError(f"Invalid authorized risk fraction: {auth_risk}")

            # Verify audit lineage presence
            if not getattr(authorization, "authorization_id", None) or \
               not getattr(authorization, "source_model_version", None) or \
               not getattr(authorization, "policy_version", None):
                raise AuthorizationError("Missing required audit lineage fields in authorization result")

            # Check expiration/stale authorization if configured
            if self.policy.authorization_max_age_seconds > 0.0:
                try:
                    context_dt = datetime.fromisoformat(context.timestamp.replace("Z", "+00:00"))
                    auth_dt = datetime.fromisoformat(authorization.evaluated_at.replace("Z", "+00:00"))
                    age = (context_dt - auth_dt).total_seconds()
                    if age > self.policy.authorization_max_age_seconds:
                        raise AuthorizationError(f"Authorization is stale: {age}s age")
                except (ValueError, TypeError) as e:
                    raise AuthorizationError(f"Audit timestamp check failed: {str(e)}")

            # 2. Market Data Freshness
            try:
                context_dt = datetime.fromisoformat(context.timestamp.replace("Z", "+00:00"))
                market_dt = datetime.fromisoformat(context.market_timestamp.replace("Z", "+00:00"))
            except (ValueError, TypeError) as e:
                raise PositionSizingValidationError(f"Invalid timestamp formatting: {str(e)}")

            age_seconds = (context_dt - market_dt).total_seconds()
            
            # Future timestamp check
            if age_seconds < -1.0:  # Allow 1s clock draft toleration
                raise PositionSizingValidationError(
                    f"Future-invalid market timestamp: market={context.market_timestamp} vs context={context.timestamp}"
                )

            # Wall clock future check
            now_utc = datetime.now(timezone.utc)
            if (context_dt - now_utc).total_seconds() > 60.0:
                raise PositionSizingValidationError(f"Execution timestamp is in the future: {context.timestamp}")

            if self.policy.reject_if_market_data_stale:
                if age_seconds > self.policy.market_data_max_age_seconds:
                    raise PositionSizingValidationError(
                        f"Stale market data: age is {age_seconds:.2f}s, max allowed {self.policy.market_data_max_age_seconds}s"
                    )

            # 3. Direction and Stop-Loss Validation
            direction = proposal.direction.upper()
            if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                raise PositionSizingValidationError(f"Unknown proposal direction: {proposal.direction}")

            if direction == "NEUTRAL":
                raise PositionSizingValidationError("NEUTRAL direction cannot trade")

            if context.stop_loss_price <= 0.0:
                raise PositionSizingValidationError("Stop-loss price must be strictly positive")

            stop_dist_abs = abs(context.entry_price - context.stop_loss_price)
            if stop_dist_abs == 0.0:
                raise InvalidStopDistanceError("Stop-loss distance cannot be zero")

            # Check stop-loss orientation
            if direction == "BULLISH":
                if context.stop_loss_price >= context.entry_price:
                    raise InvalidStopDistanceError(
                        f"BULLISH stop-loss price ({context.stop_loss_price}) must be less than entry price ({context.entry_price})"
                    )
            elif direction == "BEARISH":
                if context.stop_loss_price <= context.entry_price:
                    raise InvalidStopDistanceError(
                        f"BEARISH stop-loss price ({context.stop_loss_price}) must be greater than entry price ({context.entry_price})"
                    )

            # 4. Unsupported Instrument type check
            if context.instrument_type.lower() not in ("spot", "linear_perpetual"):
                raise PositionSizingValidationError(f"Unsupported instrument type: {context.instrument_type}")

            # 5. Core Sizing Formula
            # risk_amount = equity * authorized_risk_fraction (Leverage does not expand this!)
            risk_amount = context.equity * auth_risk
            stop_dist_frac = stop_dist_abs / context.entry_price

            # Sizing assumes: position_notional = risk_amount / stop_distance_fraction
            # quantity_raw = position_notional / (entry_price * contract_size)
            raw_qty = risk_amount / (stop_dist_frac * context.entry_price * context.contract_size)

            # 6. Apply Sizing Rounding & Step Normalization (Using Decimal to avoid precision bugs)
            dec_qty = Decimal(str(raw_qty))
            dec_step = Decimal(str(context.quantity_step))
            
            if self.policy.rounding_mode == "DOWN":
                rounded_qty = (dec_qty // dec_step) * dec_step
            elif self.policy.rounding_mode == "UP":
                rounded_qty = Decimal(math.ceil(float(dec_qty / dec_step))) * dec_step
            else:  # ROUND
                rounded_qty = Decimal(round(float(dec_qty / dec_step))) * dec_step

            quantity_normalized = float(rounded_qty)

            # 7. Recalculate economic parameters after normalisation
            actual_notional = quantity_normalized * context.entry_price * context.contract_size
            actual_risk_amount = actual_notional * stop_dist_frac
            actual_risk_fraction = actual_risk_amount / context.equity
            estimated_margin = actual_notional / context.leverage

            # Hard Check: Normalized risk must never exceed authorized risk budget!
            # Since ROUND_UP or float representation might cause it, we check fraction
            if actual_risk_fraction > (auth_risk + 1e-9):
                raise PositionLimitError(
                    f"Normalized risk fraction ({actual_risk_fraction:.6f}) exceeds authorized limit ({auth_risk:.6f})"
                )

            # 8. Exposure & Capital safety gates
            # context level checks
            if quantity_normalized < context.min_quantity or quantity_normalized > context.max_quantity:
                raise PositionLimitError(
                    f"Quantity {quantity_normalized} is outside instrument context bounds: min={context.min_quantity}, max={context.max_quantity}"
                )

            # policy level checks
            if self.policy.reject_if_below_min_quantity and quantity_normalized < self.policy.minimum_quantity:
                raise PositionLimitError(
                    f"Quantity {quantity_normalized} is below policy minimum_quantity: {self.policy.minimum_quantity}"
                )
            if self.policy.reject_if_above_max_quantity and quantity_normalized > self.policy.maximum_quantity:
                raise PositionLimitError(
                    f"Quantity {quantity_normalized} is above policy maximum_quantity: {self.policy.maximum_quantity}"
                )

            if actual_notional < self.policy.minimum_position_notional:
                raise PositionLimitError(
                    f"Position notional {actual_notional:.2f} is below policy minimum: {self.policy.minimum_position_notional}"
                )
            if actual_notional > self.policy.maximum_position_notional:
                raise PositionLimitError(
                    f"Position notional {actual_notional:.2f} is above policy maximum: {self.policy.maximum_position_notional}"
                )

            # Leverage constraint
            if context.leverage > self.policy.maximum_leverage:
                raise PositionLimitError(
                    f"Requested leverage {context.leverage} exceeds maximum allowed: {self.policy.maximum_leverage}"
                )

            # Margin fraction check
            max_margin_allowed = context.available_balance * self.policy.maximum_margin_fraction
            if estimated_margin > max_margin_allowed:
                raise InsufficientCapitalError(
                    f"Estimated margin required {estimated_margin:.2f} exceeds policy maximum margin fraction allocation {max_margin_allowed:.2f}"
                )
            
            # Absolute balance check
            if estimated_margin > context.available_balance:
                raise InsufficientCapitalError(
                    f"Estimated margin required {estimated_margin:.2f} exceeds absolute available balance {context.available_balance:.2f}"
                )

            # Symbol Exposure check
            new_sym_exposure = context.current_symbol_exposure + actual_notional
            max_sym_exp = context.equity * self.policy.maximum_symbol_exposure_fraction
            if Spec_Symbol_Exceeded := new_sym_exposure > max_sym_exp:
                raise PositionLimitError(
                    f"Symbol exposure ({new_sym_exposure:.2f}) exceeds policy maximum exposure fraction allocation ({max_sym_exp:.2f})"
                )

            # Portfolio Exposure check
            new_port_exposure = context.current_portfolio_exposure + actual_notional
            max_port_exp = context.equity * self.policy.maximum_portfolio_exposure_fraction
            if Spec_Port_Exceeded := new_port_exposure > max_port_exp:
                raise PositionLimitError(
                    f"Portfolio exposure ({new_port_exposure:.2f}) exceeds policy maximum exposure fraction allocation ({max_port_exp:.2f})"
                )

            # 9. Assembly of successful result
            success = True
            result = PositionSizeResult(
                sizing_id=str(uuid.uuid4()),
                proposal_id=proposal.proposal_id,
                symbol=context.symbol,
                direction=proposal.direction,
                quantity=quantity_normalized,
                position_notional=actual_notional,
                entry_price=context.entry_price,
                stop_loss_price=context.stop_loss_price,
                stop_distance_absolute=stop_dist_abs,
                stop_distance_fraction=stop_dist_frac,
                authorized_risk_fraction=auth_risk,
                risk_amount=actual_risk_amount,
                leverage=context.leverage,
                estimated_margin_required=estimated_margin,
                policy_version=self.policy.policy_version,
                created_at=datetime.now(timezone.utc).isoformat(),
                authorization_id=authorization.authorization_id,
                source_model_version=proposal.source_model_version,
                metadata={},
            )

            return result

        except Exception as e:
            rejection_reason = str(e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if self.telemetry_sink:
                try:
                    self.telemetry_sink.record(
                        result=result,
                        context=context,
                        success=success,
                        rejection_reason=rejection_reason,
                        latency_ms=latency_ms,
                    )
                except Exception:
                    pass  # Telemetry failures must not block positioning execution (fail-closed integrity)
