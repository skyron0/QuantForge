from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from backend.portfolio.models import PositionSide
from backend.execution_authorization.models import OrderDirection


class PositionLifecycleStatus(Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class ProtectiveTriggerType(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL_EXIT = "MANUAL_EXIT"


class ExitReason(Enum):
    STOP_LOSS_TRIGGERED = "STOP_LOSS_TRIGGERED"
    TAKE_PROFIT_TRIGGERED = "TAKE_PROFIT_TRIGGERED"
    TRAILING_STOP_TRIGGERED = "TRAILING_STOP_TRIGGERED"
    MANUAL_EXIT = "MANUAL_EXIT"
    RISK_EXIT = "RISK_EXIT"


@dataclass(frozen=True)
class ProtectivePositionState:
    lifecycle_id: str
    position_id: str
    symbol: str
    side: PositionSide
    quantity: Decimal
    average_entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    trailing_stop_enabled: bool
    trailing_distance: Optional[Decimal]
    trailing_activation_price: Optional[Decimal]
    highest_price_since_entry: Optional[Decimal]
    lowest_price_since_entry: Optional[Decimal]
    active_trailing_stop_price: Optional[Decimal]
    status: PositionLifecycleStatus
    created_at: str
    updated_at: str
    policy_version: str
    source_proposal_id: Optional[str] = None
    source_execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Fail closed validations
        if not self.lifecycle_id or not self.position_id or not self.symbol:
            raise ValueError("Identifier fields cannot be empty")
        if not isinstance(self.side, PositionSide):
            raise ValueError(f"Invalid side type: {self.side}")
        if not isinstance(self.status, PositionLifecycleStatus):
            raise ValueError(f"Invalid status type: {self.status}")
        
        # Verify decimal bounds
        if self.quantity <= Decimal("0") or self.quantity.is_nan() or self.quantity.is_infinite():
            raise ValueError(f"Invalid quantity: {self.quantity}")
        if self.average_entry_price <= Decimal("0") or self.average_entry_price.is_nan() or self.average_entry_price.is_infinite():
            raise ValueError(f"Invalid entry price: {self.average_entry_price}")
        
        for price_field in [self.stop_loss, self.take_profit, self.trailing_distance, 
                            self.trailing_activation_price, self.highest_price_since_entry, 
                            self.lowest_price_since_entry, self.active_trailing_stop_price]:
            if price_field is not None:
                if price_field.is_nan() or price_field.is_infinite():
                    raise ValueError("Price values cannot be NaN or Infinite")
                if price_field < Decimal("0"):
                    raise ValueError("Price/distance fields cannot be negative")


@dataclass(frozen=True)
class ExitProposal:
    exit_proposal_id: str
    lifecycle_id: str
    position_id: str
    symbol: str
    position_side: PositionSide
    exit_direction: OrderDirection
    requested_quantity: Decimal
    trigger_type: ProtectiveTriggerType
    exit_reason: ExitReason
    trigger_price: Decimal
    market_price: Decimal
    source_stop_loss: Optional[Decimal]
    source_take_profit: Optional[Decimal]
    source_trailing_stop: Optional[Decimal]
    created_at: str
    expires_at: str
    lifecycle_policy_version: str
    source_execution_id: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Exclude execute() or broker submission dependencies by having NO executable methods.
        # Strict validation
        if not self.exit_proposal_id or not self.lifecycle_id or not self.position_id:
            raise ValueError("All identifiers must be set")
        if not isinstance(self.position_side, PositionSide):
            raise ValueError("Invalid side")
        if not isinstance(self.exit_direction, OrderDirection):
            raise ValueError("Invalid exit direction")
        if not isinstance(self.trigger_type, ProtectiveTriggerType):
            raise ValueError("Invalid trigger type")
        if not isinstance(self.exit_reason, ExitReason):
            raise ValueError("Invalid exit reason")
            
        if self.requested_quantity <= Decimal("0") or self.requested_quantity.is_nan() or self.requested_quantity.is_infinite():
            raise ValueError("Invalid requested quantity")
        if self.trigger_price <= Decimal("0") or self.trigger_price.is_nan() or self.trigger_price.is_infinite():
            raise ValueError("Invalid trigger price")
        if self.market_price <= Decimal("0") or self.market_price.is_nan() or self.market_price.is_infinite():
            raise ValueError("Invalid market price")
