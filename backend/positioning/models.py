import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from backend.positioning.exceptions import PositionSizingValidationError


@dataclass(frozen=True)
class PositionSizingContext:
    symbol: str
    instrument_type: str  # e.g., "spot", "linear_perpetual"
    equity: float
    available_balance: float
    entry_price: float
    stop_loss_price: float
    market_price: float
    leverage: float
    contract_size: float
    lot_size: float
    min_quantity: float
    max_quantity: float
    quantity_step: float
    price_tick: float
    current_symbol_exposure: float
    current_portfolio_exposure: float
    market_timestamp: str  # ISO 8601
    timestamp: str  # ISO 8601
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # String validations
        if not self.symbol:
            raise PositionSizingValidationError("Symbol cannot be empty")
        if not self.instrument_type:
            raise PositionSizingValidationError("Instrument type cannot be empty")

        # Helper to validate float fields
        def _val(val: Any, name: str, min_val: Optional[float] = 0.0, allow_zero: bool = True):
            if not isinstance(val, (int, float)):
                raise PositionSizingValidationError(f"{name} must be a numeric value")
            if math.isnan(val) or math.isinf(val):
                raise PositionSizingValidationError(f"{name} cannot be NaN or Inf")
            if min_val is not None:
                if allow_zero:
                    if val < min_val:
                        raise PositionSizingValidationError(f"{name} ({val}) cannot be less than {min_val}")
                else:
                    if val <= min_val:
                        raise PositionSizingValidationError(f"{name} ({val}) must be strictly greater than {min_val}")

        # Core numeric validations
        _val(self.equity, "equity", min_val=0.0, allow_zero=False)
        _val(self.available_balance, "available_balance", min_val=0.0, allow_zero=True)
        _val(self.entry_price, "entry_price", min_val=0.0, allow_zero=False)
        _val(self.stop_loss_price, "stop_loss_price", min_val=0.0, allow_zero=False)
        _val(self.market_price, "market_price", min_val=0.0, allow_zero=False)
        _val(self.leverage, "leverage", min_val=0.0, allow_zero=False)
        _val(self.contract_size, "contract_size", min_val=0.0, allow_zero=False)
        _val(self.lot_size, "lot_size", min_val=0.0, allow_zero=False)
        _val(self.min_quantity, "min_quantity", min_val=0.0, allow_zero=False)
        _val(self.max_quantity, "max_quantity", min_val=0.0, allow_zero=False)
        _val(self.quantity_step, "quantity_step", min_val=0.0, allow_zero=False)
        _val(self.price_tick, "price_tick", min_val=0.0, allow_zero=False)
        _val(self.current_symbol_exposure, "current_symbol_exposure", min_val=0.0, allow_zero=True)
        _val(self.current_portfolio_exposure, "current_portfolio_exposure", min_val=0.0, allow_zero=True)

        if self.min_quantity > self.max_quantity:
            raise PositionSizingValidationError(
                f"min_quantity ({self.min_quantity}) cannot exceed max_quantity ({self.max_quantity})"
            )

        # Check ISO timestamp format
        try:
            datetime.fromisoformat(self.market_timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise PositionSizingValidationError(f"Invalid market_timestamp format: {self.market_timestamp}")

        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise PositionSizingValidationError(f"Invalid execution timestamp format: {self.timestamp}")


@dataclass(frozen=True)
class PositionSizeResult:
    sizing_id: str
    proposal_id: str
    symbol: str
    direction: str
    quantity: float
    position_notional: float
    entry_price: float
    stop_loss_price: float
    stop_distance_absolute: float
    stop_distance_fraction: float
    authorized_risk_fraction: float
    risk_amount: float
    leverage: float
    estimated_margin_required: float
    policy_version: str
    created_at: str
    authorization_id: Optional[str]
    source_model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
