import math
from dataclasses import dataclass

@dataclass(frozen=True)
class PaperExecutionPolicy:
    policy_version: str
    maximum_market_data_age_seconds: float = 10.0
    maximum_future_clock_skew_seconds: float = 2.0
    fee_rate: float = 0.001  # 0.1% fee
    slippage_rate: float = 0.0005  # 0.05% slippage
    allow_partial_fills: bool = True
    minimum_fill_quantity: float = 0.0001
    reject_if_insufficient_liquidity: bool = False
    intent_max_age_seconds: float = 60.0
    execution_result_ttl_seconds: float = 3600.0

    def __post_init__(self):
        if not self.policy_version:
            raise ValueError("policy_version cannot be empty")
        for name, val in [
            ("maximum_market_data_age_seconds", self.maximum_market_data_age_seconds),
            ("maximum_future_clock_skew_seconds", self.maximum_future_clock_skew_seconds),
            ("fee_rate", self.fee_rate),
            ("slippage_rate", self.slippage_rate),
            ("minimum_fill_quantity", self.minimum_fill_quantity),
            ("intent_max_age_seconds", self.intent_max_age_seconds),
            ("execution_result_ttl_seconds", self.execution_result_ttl_seconds),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"{name} must be a finite number")
            if val < 0:
                raise ValueError(f"{name} cannot be negative")

        if self.fee_rate > 1.0:
            raise ValueError("fee_rate cannot exceed 1.0 (100%)")
        if self.slippage_rate > 1.0:
            raise ValueError("slippage_rate cannot exceed 1.0 (100%)")
