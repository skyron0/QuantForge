import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from backend.positioning.exceptions import InvalidPositionSizingPolicyError


@dataclass(frozen=True)
class PositionSizingPolicy:
    policy_version: str
    minimum_position_notional: float
    maximum_position_notional: float
    minimum_quantity: float
    maximum_quantity: float
    maximum_leverage: float
    maximum_margin_fraction: float
    maximum_symbol_exposure_fraction: float
    maximum_portfolio_exposure_fraction: float
    rounding_mode: str  # "DOWN", "UP", "ROUND"
    reject_if_below_min_quantity: bool
    reject_if_above_max_quantity: bool
    reject_if_stop_distance_invalid: bool
    reject_if_market_data_stale: bool
    market_data_max_age_seconds: float
    authorization_max_age_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.policy_version:
            raise InvalidPositionSizingPolicyError("policy_version cannot be empty")

        # Validation helper
        def _val(val: Any, name: str, min_val: Optional[float] = 0.0, max_val: Optional[float] = None):
            if not isinstance(val, (int, float)):
                raise InvalidPositionSizingPolicyError(f"{name} must be numeric")
            if math.isnan(val) or math.isinf(val):
                raise InvalidPositionSizingPolicyError(f"{name} cannot be NaN or Inf")
            if min_val is not None and val < min_val:
                raise InvalidPositionSizingPolicyError(f"{name} cannot be less than {min_val}")
            if max_val is not None and val > max_val:
                raise InvalidPositionSizingPolicyError(f"{name} cannot be greater than {max_val}")

        _val(self.minimum_position_notional, "minimum_position_notional", min_val=0.0)
        _val(self.maximum_position_notional, "maximum_position_notional", min_val=0.0)
        _val(self.minimum_quantity, "minimum_quantity", min_val=0.0)
        _val(self.maximum_quantity, "maximum_quantity", min_val=0.0)
        _val(self.maximum_leverage, "maximum_leverage", min_val=0.0001)
        _val(self.maximum_margin_fraction, "maximum_margin_fraction", min_val=0.0, max_val=1.0)
        _val(self.maximum_symbol_exposure_fraction, "maximum_symbol_exposure_fraction", min_val=0.0, max_val=1.0)
        _val(self.maximum_portfolio_exposure_fraction, "maximum_portfolio_exposure_fraction", min_val=0.0, max_val=1.0)
        _val(self.market_data_max_age_seconds, "market_data_max_age_seconds", min_val=0.0)
        _val(self.authorization_max_age_seconds, "authorization_max_age_seconds", min_val=0.0)

        if self.minimum_position_notional > self.maximum_position_notional:
            raise InvalidPositionSizingPolicyError(
                f"minimum_position_notional ({self.minimum_position_notional}) cannot exceed maximum_position_notional ({self.maximum_position_notional})"
            )

        if self.minimum_quantity > self.maximum_quantity:
            raise InvalidPositionSizingPolicyError(
                f"minimum_quantity ({self.minimum_quantity}) cannot exceed maximum_quantity ({self.maximum_quantity})"
            )

        if self.rounding_mode not in {"DOWN", "UP", "ROUND"}:
            raise InvalidPositionSizingPolicyError(
                f"Invalid rounding_mode '{self.rounding_mode}'. Must be one of DOWN, UP, ROUND."
            )
