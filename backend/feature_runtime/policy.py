"""
Versioned configuration policy for the Feature Runtime.
"""

from dataclasses import dataclass
from backend.feature_runtime.exceptions import InvalidFeaturePolicyError


@dataclass(frozen=True)
class FeatureRuntimePolicy:
    """
    Deterministic, self-validating policy governing the Feature Runtime.
    """

    policy_version: str = "feature-runtime-v1"

    # Minimum number of historical candles before features can be extracted.
    minimum_history: int = 100

    # Maximum age (seconds) before market data is considered stale.
    staleness_limit_seconds: float = 10.0

    # Maximum feature buffer size (per symbol).
    buffer_capacity: int = 500

    # Whether the feature runtime is enabled.
    enabled: bool = True

    # Model version identifier (used for inference routing).
    model_version: str = "default-model-v1"

    # Timeframe label for the generated MLSignal.
    default_timeframe: str = "5m"

    # Confidence thresholds for signal mapping.
    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45

    def __post_init__(self) -> None:
        if self.minimum_history < 1:
            raise InvalidFeaturePolicyError(
                f"minimum_history must be >= 1, got {self.minimum_history}"
            )
        if self.staleness_limit_seconds <= 0:
            raise InvalidFeaturePolicyError(
                f"staleness_limit_seconds must be > 0, got {self.staleness_limit_seconds}"
            )
        if self.buffer_capacity < self.minimum_history:
            raise InvalidFeaturePolicyError(
                f"buffer_capacity ({self.buffer_capacity}) must be >= "
                f"minimum_history ({self.minimum_history})"
            )
        if not (0.0 < self.bearish_threshold < self.bullish_threshold < 1.0):
            raise InvalidFeaturePolicyError(
                f"Thresholds must satisfy 0 < bearish ({self.bearish_threshold}) "
                f"< bullish ({self.bullish_threshold}) < 1"
            )
