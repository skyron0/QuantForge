import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from backend.execution_authorization.exceptions import ExecutionValidationError


class ExecutionEnvironment(str, Enum):
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


class OrderDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ExecutionAuthorizationStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExecutionContext:
    environment: ExecutionEnvironment
    current_timestamp: str
    market_timestamp: str
    execution_enabled: bool
    kill_switch_active: bool
    symbol_trading_enabled: bool
    available_balance: float
    current_price: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate environment type
        if not isinstance(self.environment, ExecutionEnvironment):
            raise ExecutionValidationError(f"Invalid environment: {self.environment}")

        # Validate numeric inputs
        for field_name in ["available_balance", "current_price"]:
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)):
                raise ExecutionValidationError(f"Field {field_name} must be a number, got {type(val)}")
            if math.isnan(val):
                raise ExecutionValidationError(f"Field {field_name} must not be NaN")
            if math.isinf(val):
                raise ExecutionValidationError(f"Field {field_name} must not be Infinite")
            if val < 0:
                raise ExecutionValidationError(f"Field {field_name} must not be negative: {val}")


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    idempotency_key: str
    proposal_id: str
    risk_authorization_id: str
    sizing_id: str
    symbol: str
    direction: OrderDirection
    quantity: float
    order_type: OrderType
    limit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    environment: ExecutionEnvironment
    source_model_version: str
    fusion_policy_version: str
    risk_policy_version: str
    position_sizing_policy_version: str
    execution_policy_version: str
    reasoning_request_id: Optional[str]
    created_at: str
    expires_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Enforce direction and type enum types
        if not isinstance(self.direction, OrderDirection):
            raise ExecutionValidationError(f"Invalid direction: {self.direction}")
        if not isinstance(self.order_type, OrderType):
            raise ExecutionValidationError(f"Invalid order_type: {self.order_type}")
        if not isinstance(self.environment, ExecutionEnvironment):
            raise ExecutionValidationError(f"Invalid environment: {self.environment}")

        # Verify quantities and prices
        for num_name in ["quantity", "limit_price", "stop_loss", "take_profit"]:
            val = getattr(self, num_name)
            if val is not None:
                if not isinstance(val, (int, float)):
                    raise ExecutionValidationError(f"{num_name} must be a number, got {type(val)}")
                if math.isnan(val):
                    raise ExecutionValidationError(f"{num_name} must not be NaN")
                if math.isinf(val):
                    raise ExecutionValidationError(f"{num_name} must not be Infinite")
                if val <= 0:
                    raise ExecutionValidationError(f"{num_name} must be greater than zero: {val}")


@dataclass(frozen=True)
class ExecutionAuthorizationResult:
    authorization_id: str
    status: ExecutionAuthorizationStatus
    intent: Optional[OrderIntent]
    rejection_reason: str
    triggered_rules: List[str]
    policy_version: str
    proposal_id: str
    risk_authorization_id: str
    sizing_id: str
    latency_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.status, ExecutionAuthorizationStatus):
            raise ExecutionValidationError(f"Invalid authorization status: {self.status}")
        if self.status == ExecutionAuthorizationStatus.AUTHORIZED and self.intent is None:
            raise ExecutionValidationError("AUTHORIZED state must contain an OrderIntent")
        if self.status == ExecutionAuthorizationStatus.REJECTED and self.intent is not None:
            raise ExecutionValidationError("REJECTED state must not contain an OrderIntent")
