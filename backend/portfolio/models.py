from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Mapping

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass(frozen=True)
class Position:
    position_id: str
    symbol: str
    side: PositionSide
    quantity: Decimal
    average_entry_price: Decimal
    current_price: Decimal
    position_notional: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    accumulated_fees: Decimal
    leverage: Decimal
    margin_used: Decimal
    opened_at: str
    updated_at: str
    source_execution_ids: List[str] = field(default_factory=list)
    source_fill_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate no NaN/Inf
        for name, value in [
            ("quantity", self.quantity),
            ("average_entry_price", self.average_entry_price),
            ("current_price", self.current_price),
            ("position_notional", self.position_notional),
            ("unrealized_pnl", self.unrealized_pnl),
            ("realized_pnl", self.realized_pnl),
            ("accumulated_fees", self.accumulated_fees),
            ("leverage", self.leverage),
            ("margin_used", self.margin_used)
        ]:
            if not isinstance(value, Decimal):
                raise TypeError(f"Field '{name}' must be of type Decimal, got {type(value)}")
            if value.is_nan() or value.is_infinite():
                raise ValueError(f"Field '{name}' cannot be NaN or Infinite: {value}")
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Position quantity must be positive, got {self.quantity}")
        if self.average_entry_price <= Decimal("0"):
            raise ValueError(f"Average entry price must be positive, got {self.average_entry_price}")
        if self.current_price <= Decimal("0"):
            raise ValueError(f"Current price must be positive, got {self.current_price}")
        if self.leverage <= Decimal("0"):
            raise ValueError(f"Leverage must be positive, got {self.leverage}")
        if self.margin_used < Decimal("0"):
            raise ValueError(f"Margin used cannot be negative, got {self.margin_used}")

@dataclass(frozen=True)
class PortfolioState:
    portfolio_id: str
    initial_balance: Decimal
    cash_balance: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    used_margin: Decimal
    available_balance: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    open_position_count: int
    positions: Mapping[str, Position] = field(default_factory=dict)  # Maps symbol -> Position
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate no NaN/Inf
        for name, value in [
            ("initial_balance", self.initial_balance),
            ("cash_balance", self.cash_balance),
            ("equity", self.equity),
            ("realized_pnl", self.realized_pnl),
            ("unrealized_pnl", self.unrealized_pnl),
            ("total_fees", self.total_fees),
            ("used_margin", self.used_margin),
            ("available_balance", self.available_balance),
            ("gross_exposure", self.gross_exposure),
            ("net_exposure", self.net_exposure)
        ]:
            if not isinstance(value, Decimal):
                raise TypeError(f"Field '{name}' must be of type Decimal, got {type(value)}")
            if value.is_nan() or value.is_infinite():
                raise ValueError(f"Field '{name}' cannot be NaN or Infinite: {value}")
        if self.initial_balance <= Decimal("0"):
            raise ValueError(f"Initial balance must be positive, got {self.initial_balance}")
        if self.used_margin < Decimal("0"):
            raise ValueError(f"Used margin cannot be negative, got {self.used_margin}")
        if self.total_fees < Decimal("0"):
            raise ValueError(f"Total fees cannot be negative, got {self.total_fees}")
        if self.gross_exposure < Decimal("0"):
            raise ValueError(f"Gross exposure cannot be negative, got {self.gross_exposure}")
        if abs(self.net_exposure) > self.gross_exposure:
            raise ValueError(
                f"Gross exposure ({self.gross_exposure}) must be greater than or equal to absolute net exposure ({abs(self.net_exposure)})"
            )
        if not isinstance(self.positions, MappingProxyType):
            object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))

@dataclass(frozen=True)
class PortfolioSnapshot:
    portfolio_id: str
    initial_balance: Decimal
    cash_balance: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    used_margin: Decimal
    available_balance: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    open_positions: List[Position] = field(default_factory=list)
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
