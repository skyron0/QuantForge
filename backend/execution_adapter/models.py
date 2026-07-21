import math
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from backend.execution_authorization.models import ExecutionEnvironment, OrderDirection, OrderType

class ExecutionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

@dataclass(frozen=True)
class Fill:
    fill_id: str
    intent_id: str
    symbol: str
    direction: OrderDirection
    quantity: float
    price: float
    notional: float
    fee: float
    slippage_amount: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.fill_id:
            raise ValueError("fill_id cannot be empty")
        if not self.intent_id:
            raise ValueError("intent_id cannot be empty")
        if math.isnan(self.quantity) or math.isinf(self.quantity) or self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}")
        if math.isnan(self.price) or math.isinf(self.price) or self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if math.isnan(self.notional) or math.isinf(self.notional) or self.notional <= 0:
            raise ValueError(f"Invalid notional: {self.notional}")
        if math.isnan(self.fee) or math.isinf(self.fee) or self.fee < 0:
            raise ValueError(f"Invalid fee: {self.fee}")
        if math.isnan(self.slippage_amount) or math.isinf(self.slippage_amount) or self.slippage_amount < 0:
            raise ValueError(f"Invalid slippage_amount: {self.slippage_amount}")

@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    intent_id: str
    proposal_id: str
    risk_authorization_id: str
    sizing_id: str
    symbol: str
    direction: OrderDirection
    requested_quantity: float
    filled_quantity: float
    average_fill_price: float
    total_notional: float
    total_fees: float
    total_slippage: float
    status: ExecutionStatus
    fills: List[Fill] = field(default_factory=list)
    rejection_reason: str = ""
    adapter_name: str = ""
    environment: ExecutionEnvironment = ExecutionEnvironment.PAPER
    started_at: str = ""
    completed_at: str = ""
    latency_ms: float = 0.0
    policy_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Verification of numerical bounds
        if math.isnan(self.requested_quantity) or math.isinf(self.requested_quantity) or self.requested_quantity <= 0:
            raise ValueError(f"Invalid requested_quantity: {self.requested_quantity}")
        if math.isnan(self.filled_quantity) or math.isinf(self.filled_quantity) or self.filled_quantity < 0:
            raise ValueError(f"Invalid filled_quantity: {self.filled_quantity}")
        if self.filled_quantity > self.requested_quantity:
            raise ValueError(f"filled_quantity ({self.filled_quantity}) cannot exceed requested_quantity ({self.requested_quantity})")
        if math.isnan(self.average_fill_price) or math.isinf(self.average_fill_price) or self.average_fill_price < 0:
            raise ValueError(f"Invalid average_fill_price: {self.average_fill_price}")
        if math.isnan(self.total_notional) or math.isinf(self.total_notional) or self.total_notional < 0:
            raise ValueError(f"Invalid total_notional: {self.total_notional}")
        if math.isnan(self.total_fees) or math.isinf(self.total_fees) or self.total_fees < 0:
            raise ValueError(f"Invalid total_fees: {self.total_fees}")
        if math.isnan(self.total_slippage) or math.isinf(self.total_slippage) or self.total_slippage < 0:
            raise ValueError(f"Invalid total_slippage: {self.total_slippage}")

        # Invariant validations
        if self.status == ExecutionStatus.FILLED:
            if not math.isclose(self.filled_quantity, self.requested_quantity, rel_tol=1e-9):
                raise ValueError(f"FILLED status requires filled_quantity == requested_quantity. Filled: {self.filled_quantity}, Requested: {self.requested_quantity}")
        elif self.status == ExecutionStatus.PARTIALLY_FILLED:
            if not (0 < self.filled_quantity < self.requested_quantity):
                raise ValueError(f"PARTIALLY_FILLED status requires 0 < filled_quantity < requested_quantity. Filled: {self.filled_quantity}, Requested: {self.requested_quantity}")
        elif self.status == ExecutionStatus.REJECTED:
            if self.filled_quantity != 0:
                raise ValueError(f"REJECTED status requires filled_quantity == 0. Filled: {self.filled_quantity}")
        elif self.status == ExecutionStatus.ACCEPTED:
            if self.filled_quantity != 0:
                raise ValueError(f"ACCEPTED status requires filled_quantity == 0. Filled: {self.filled_quantity}")

@dataclass(frozen=True)
class PaperExecutionContext:
    current_market_price: float
    bid_price: float
    ask_price: float
    available_liquidity: float
    timestamp: str  # ISO 8601
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate all numeric values
        for k, v in [
            ("current_market_price", self.current_market_price),
            ("bid_price", self.bid_price),
            ("ask_price", self.ask_price),
            ("available_liquidity", self.available_liquidity),
        ]:
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"{k} must be a finite float")
            if v < 0:
                raise ValueError(f"{k} cannot be negative")

        if self.current_market_price <= 0:
            raise ValueError("current_market_price must be greater than zero")
        if self.bid_price <= 0:
            raise ValueError("bid_price must be greater than zero")
        if self.ask_price <= 0:
            raise ValueError("ask_price must be greater than zero")
        if self.bid_price > self.ask_price:
            raise ValueError(f"bid_price ({self.bid_price}) cannot be greater than ask_price ({self.ask_price})")
        if not self.timestamp:
            raise ValueError("timestamp must not be empty or null")
