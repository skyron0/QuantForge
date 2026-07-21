from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional
from backend.position_lifecycle.exceptions import InvalidLifecyclePolicyError
from backend.position_lifecycle.models import ProtectiveTriggerType


@dataclass(frozen=True)
class PositionLifecyclePolicy:
    policy_version: str
    allow_stop_loss: bool
    require_stop_loss: bool
    allow_take_profit: bool
    require_take_profit: bool
    allow_trailing_stop: bool
    minimum_stop_distance_fraction: Decimal
    maximum_stop_distance_fraction: Decimal
    minimum_take_profit_distance_fraction: Decimal
    trailing_distance_mode: str  # "ABSOLUTE" or "PERCENTAGE"
    minimum_trailing_distance: Decimal
    maximum_trailing_distance: Decimal
    trigger_priority: List[ProtectiveTriggerType] = field(
        default_factory=lambda: [
            ProtectiveTriggerType.STOP_LOSS,
            ProtectiveTriggerType.TAKE_PROFIT,
            ProtectiveTriggerType.TRAILING_STOP
        ]
    )
    exit_proposal_ttl_seconds: float = 60.0
    maximum_market_data_age_seconds: float = 60.0
    maximum_future_clock_skew_seconds: float = 5.0
    allow_breakeven: bool = False
    breakeven_activation_fraction: Optional[Decimal] = None
    breakeven_offset_fraction: Optional[Decimal] = None

    def __post_init__(self):
        # Strict validation
        if not self.policy_version:
            raise InvalidLifecyclePolicyError("policy_version cannot be empty")
        
        # Cross-field checks
        if self.require_stop_loss and not self.allow_stop_loss:
            raise InvalidLifecyclePolicyError("Cannot require SL when SL is not allowed")
        if self.require_take_profit and not self.allow_take_profit:
            raise InvalidLifecyclePolicyError("Cannot require TP when TP is not allowed")
            
        if self.minimum_stop_distance_fraction < Decimal("0") or self.maximum_stop_distance_fraction < Decimal("0"):
            raise InvalidLifecyclePolicyError("Stop distance fraction cannot be negative")
        if self.minimum_stop_distance_fraction > self.maximum_stop_distance_fraction:
            raise InvalidLifecyclePolicyError("minimum_stop_distance_fraction cannot exceed maximum_stop_distance_fraction")
            
        if self.minimum_take_profit_distance_fraction < Decimal("0"):
            raise InvalidLifecyclePolicyError("minimum_take_profit_distance_fraction cannot be negative")
            
        if self.trailing_distance_mode not in ["ABSOLUTE", "PERCENTAGE"]:
            raise InvalidLifecyclePolicyError(f"Unsupported trailing distance mode: {self.trailing_distance_mode}")
            
        if self.minimum_trailing_distance < Decimal("0") or self.maximum_trailing_distance < Decimal("0"):
            raise InvalidLifecyclePolicyError("Trailing distance bounds cannot be negative")
        if self.minimum_trailing_distance > self.maximum_trailing_distance:
            raise InvalidLifecyclePolicyError("minimum_trailing_distance cannot exceed maximum_trailing_distance")
            
        if self.exit_proposal_ttl_seconds <= 0:
            raise InvalidLifecyclePolicyError("exit_proposal_ttl_seconds must be positive")
        if self.maximum_market_data_age_seconds <= 0:
            raise InvalidLifecyclePolicyError("maximum_market_data_age_seconds must be positive")
        if self.maximum_future_clock_skew_seconds < 0:
            raise InvalidLifecyclePolicyError("maximum_future_clock_skew_seconds cannot be negative")
            
        if not self.trigger_priority:
            raise InvalidLifecyclePolicyError("trigger_priority list cannot be empty")
        
        # Validation of trigger_priority
        for priority in self.trigger_priority:
            if not isinstance(priority, ProtectiveTriggerType):
                raise InvalidLifecyclePolicyError(f"Invalid priority trigger type: {priority}")

        # Breakeven validation
        if self.allow_breakeven:
            if self.breakeven_activation_fraction is None:
                raise InvalidLifecyclePolicyError("breakeven_activation_fraction is required if breakeven is allowed")
            if self.breakeven_offset_fraction is None:
                raise InvalidLifecyclePolicyError("breakeven_offset_fraction is required if breakeven is allowed")
            if self.breakeven_activation_fraction < Decimal("0") or self.breakeven_offset_fraction < Decimal("0"):
                raise InvalidLifecyclePolicyError("Breakeven fraction metrics must be positive/zero")
