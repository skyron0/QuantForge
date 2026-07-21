from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional
from backend.portfolio.exceptions import InvalidPortfolioPolicyError

@dataclass(frozen=True)
class PortfolioPolicy:
    policy_version: str
    supported_instrument_types: List[str]
    allow_position_reversal: bool = True
    maximum_open_positions: int = 10
    maximum_symbol_positions: int = 1
    maximum_gross_exposure_fraction: Decimal = Decimal("2.0")
    maximum_net_exposure_fraction: Decimal = Decimal("1.0")
    maximum_leverage: Decimal = Decimal("20.0")
    market_price_max_age_seconds: float = 60.0
    maximum_future_clock_skew_seconds: float = 10.0
    decimal_precision: int = 8
    accounting_tolerance: Decimal = Decimal("0.0001")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.policy_version.strip():
            raise InvalidPortfolioPolicyError("Policy version cannot be empty")
        if not isinstance(self.supported_instrument_types, list):
            raise TypeError("supported_instrument_types must be a list")
        if self.maximum_open_positions <= 0:
            raise InvalidPortfolioPolicyError("maximum_open_positions must be positive")
        if self.maximum_symbol_positions <= 0:
            raise InvalidPortfolioPolicyError("maximum_symbol_positions must be positive")
        
        for name, dec_val in [
            ("maximum_gross_exposure_fraction", self.maximum_gross_exposure_fraction),
            ("maximum_net_exposure_fraction", self.maximum_net_exposure_fraction),
            ("maximum_leverage", self.maximum_leverage),
            ("accounting_tolerance", self.accounting_tolerance)
        ]:
            if not isinstance(dec_val, Decimal):
                raise TypeError(f"Field '{name}' must be of type Decimal")
            if dec_val.is_nan() or dec_val.is_infinite():
                raise InvalidPortfolioPolicyError(f"Field '{name}' cannot be NaN or Infinite")
                
        if self.maximum_gross_exposure_fraction <= Decimal("0"):
            raise InvalidPortfolioPolicyError("maximum_gross_exposure_fraction must be positive")
        if self.maximum_net_exposure_fraction <= Decimal("0"):
            raise InvalidPortfolioPolicyError("maximum_net_exposure_fraction must be positive")
        if self.maximum_leverage <= Decimal("0"):
            raise InvalidPortfolioPolicyError("maximum_leverage must be positive")
        if self.market_price_max_age_seconds <= 0.0:
            raise InvalidPortfolioPolicyError("market_price_max_age_seconds must be positive")
        if self.maximum_future_clock_skew_seconds <= 0.0:
            raise InvalidPortfolioPolicyError("maximum_future_clock_skew_seconds must be positive")
        if self.decimal_precision <= 0:
            raise InvalidPortfolioPolicyError("decimal_precision must be positive")
        if self.accounting_tolerance < Decimal("0"):
            raise InvalidPortfolioPolicyError("accounting_tolerance cannot be negative")
