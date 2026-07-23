from dataclasses import dataclass
from typing import Optional
from backend.runtime.exceptions import PolicyValidationError


@dataclass(frozen=True)
class RuntimePolicy:
    """Configures the operational parameters and validations for the TradingRuntime."""
    scheduler_interval_seconds: float = 1.0
    max_runtime_duration_seconds: Optional[float] = None
    max_event_queue_size: int = 10000
    clock_skew_tolerance_seconds: float = 5.0
    telemetry_enabled: bool = True
    max_dispatch_latency_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.scheduler_interval_seconds <= 0.0:
            raise PolicyValidationError(
                f"scheduler_interval_seconds must be positive, got {self.scheduler_interval_seconds}"
            )
        if self.max_runtime_duration_seconds is not None and self.max_runtime_duration_seconds <= 0.0:
            raise PolicyValidationError(
                f"max_runtime_duration_seconds must be positive, got {self.max_runtime_duration_seconds}"
            )
        if self.max_event_queue_size <= 0:
            raise PolicyValidationError(
                f"max_event_queue_size must be positive, got {self.max_event_queue_size}"
            )
        if self.clock_skew_tolerance_seconds <= 0.0:
            raise PolicyValidationError(
                f"clock_skew_tolerance_seconds must be positive, got {self.clock_skew_tolerance_seconds}"
            )
        if self.max_dispatch_latency_ms <= 0.0:
            raise PolicyValidationError(
                f"max_dispatch_latency_ms must be positive, got {self.max_dispatch_latency_ms}"
            )
