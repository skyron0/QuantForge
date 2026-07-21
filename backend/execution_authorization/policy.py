from dataclasses import dataclass, field
from typing import List
from backend.execution_authorization.models import ExecutionEnvironment, OrderType
from backend.execution_authorization.exceptions import InvalidExecutionPolicyError


@dataclass(frozen=True)
class ExecutionPolicy:
    policy_version: str
    allowed_environments: List[ExecutionEnvironment]
    maximum_market_data_age_seconds: float
    order_intent_ttl_seconds: float
    minimum_quantity: float
    maximum_quantity: float
    require_stop_loss: bool
    require_take_profit: bool
    allowed_order_types: List[OrderType]
    allow_live_execution_intents: bool = False  # Set default False for safety
    require_execution_enabled: bool = True
    reject_when_kill_switch_active: bool = True
    require_symbol_enabled: bool = True
    maximum_clock_skew_seconds: float = 5.0

    def __post_init__(self):
        if not self.policy_version:
            raise InvalidExecutionPolicyError("policy_version must not be empty")

        if not self.allowed_environments:
            raise InvalidExecutionPolicyError("allowed_environments must contain at least one environment")

        for env in self.allowed_environments:
            if not isinstance(env, ExecutionEnvironment):
                raise InvalidExecutionPolicyError(f"Invalid environment in allowed_environments: {env}")

        if self.maximum_market_data_age_seconds <= 0:
            raise InvalidExecutionPolicyError("maximum_market_data_age_seconds must be positive")

        if self.order_intent_ttl_seconds <= 0:
            raise InvalidExecutionPolicyError("order_intent_ttl_seconds must be positive")

        if self.maximum_clock_skew_seconds <= 0:
            raise InvalidExecutionPolicyError("maximum_clock_skew_seconds must be positive")

        if self.minimum_quantity <= 0:
            raise InvalidExecutionPolicyError("minimum_quantity must be positive")

        if self.maximum_quantity <= 0:
            raise InvalidExecutionPolicyError("maximum_quantity must be positive")

        if self.minimum_quantity > self.maximum_quantity:
            raise InvalidExecutionPolicyError("minimum_quantity cannot exceed maximum_quantity")

        if not self.allowed_order_types:
            raise InvalidExecutionPolicyError("allowed_order_types must contain at least one order type")

        for ot in self.allowed_order_types:
            if not isinstance(ot, OrderType):
                raise InvalidExecutionPolicyError(f"Invalid order type in allowed_order_types: {ot}")
