from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import math

from backend.risk.exceptions import RiskContextError


class RiskAuthorizationStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ADJUSTED = "ADJUSTED"


@dataclass(frozen=True)
class RiskContext:
    symbol: str
    timestamp: str
    equity: float
    available_balance: float
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    current_drawdown_pct: float  # e.g., 0.05 = 5%
    portfolio_exposure_pct: float  # e.g., 0.20 = 20%
    symbol_exposure_pct: float  # e.g., 0.05 = 5%
    current_leverage: float
    open_positions_count: int
    symbol_open_positions_count: int
    volatility_state: str  # LOW, NORMAL, HIGH, CRITICAL, UNKNOWN
    consecutive_losses: int
    market_liquidity_state: str = "UNKNOWN"  # NORMAL, THIN, CRITICAL, UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Numeric validation (fail-closed if NaN/Inf)
        fields_to_check = [
            ("equity", self.equity),
            ("available_balance", self.available_balance),
            ("daily_realized_pnl", self.daily_realized_pnl),
            ("daily_unrealized_pnl", self.daily_unrealized_pnl),
            ("current_drawdown_pct", self.current_drawdown_pct),
            ("portfolio_exposure_pct", self.portfolio_exposure_pct),
            ("symbol_exposure_pct", self.symbol_exposure_pct),
            ("current_leverage", self.current_leverage),
        ]
        for name, val in fields_to_check:
            if math.isnan(val) or math.isinf(val):
                raise RiskContextError(
                    f"Invalid numeric input for risk context field {name}: {val}"
                )

        if self.equity <= 0:
            raise RiskContextError(f"Equity must be positive: {self.equity}")
        if self.current_leverage < 0:
            raise RiskContextError(f"Current leverage cannot be negative: {self.current_leverage}")
        if self.open_positions_count < 0:
            raise RiskContextError(f"Open positions count cannot be negative: {self.open_positions_count}")
        if self.symbol_open_positions_count < 0:
            raise RiskContextError(f"Symbol open positions count cannot be negative: {self.symbol_open_positions_count}")
        if self.consecutive_losses < 0:
            raise RiskContextError(f"Consecutive losses cannot be negative: {self.consecutive_losses}")
        if self.volatility_state not in ("LOW", "NORMAL", "HIGH", "CRITICAL", "UNKNOWN"):
            raise RiskContextError(f"Invalid volatility state: {self.volatility_state}")
        if self.market_liquidity_state not in ("NORMAL", "THIN", "CRITICAL", "UNKNOWN"):
            raise RiskContextError(f"Invalid market liquidity state: {self.market_liquidity_state}")


@dataclass(frozen=True)
class RiskAuthorizationResult:
    authorization_id: str
    proposal_id: str
    symbol: str
    direction: str
    status: RiskAuthorizationStatus
    original_confidence: float
    effective_confidence: float
    rejection_reasons: List[str]
    adjustment_reasons: List[str]
    triggered_rules: List[str]
    policy_version: str
    source_model_version: str
    fusion_policy_version: str
    proposal_created_at: str
    evaluated_at: str
    latency_ms: float
    requested_risk_fraction: float
    authorized_risk_fraction: float
    reasoning_request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
