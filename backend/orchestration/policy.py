from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict
from backend.orchestration.exceptions import OrchestrationValidationError


@dataclass(frozen=True)
class TradingCyclePolicy:
    policy_version: str
    enable_paper_execution: bool = True
    require_portfolio_update: bool = True
    require_lifecycle_registration: bool = True
    maximum_cycle_age_seconds: float = 60.0
    maximum_clock_skew_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Validate policy fields to ensure sensible numeric boundaries."""
        if not self.policy_version:
            raise OrchestrationValidationError("policy_version cannot be empty")
        if self.maximum_cycle_age_seconds <= 0:
            raise OrchestrationValidationError("maximum_cycle_age_seconds must be positive")
        if self.maximum_clock_skew_seconds <= 0:
            raise OrchestrationValidationError("maximum_clock_skew_seconds must be positive")
