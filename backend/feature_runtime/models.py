"""
Immutable domain models for the Feature Runtime.

FeatureSnapshot preserves explicit deterministic feature ordering via
parallel feature_names / feature_values lists — never an unordered dict.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.decision.models import MLSignal


# ─── Enums ───────────────────────────────────────────────────────────────────

class FeatureRuntimeStatus(str, Enum):
    SUCCESS = "SUCCESS"
    WARMUP_SKIP = "WARMUP_SKIP"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    STALE_DATA = "STALE_DATA"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    SIGNAL_FAILED = "SIGNAL_FAILED"
    FAILED = "FAILED"


# ─── Snapshots & Results ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSnapshot:
    """
    Ordered, immutable feature vector.

    feature_names[i] corresponds to feature_values[i].
    """

    symbol: str
    timestamp: str
    feature_names: List[str]
    feature_values: List[float]
    feature_version: str
    schema_fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.feature_values):
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) "
                f"must equal feature_values length ({len(self.feature_values)})"
            )

    def as_dict(self) -> Dict[str, float]:
        """Convenience accessor — does NOT change the canonical storage."""
        return dict(zip(self.feature_names, self.feature_values))


@dataclass(frozen=True)
class PredictionResult:
    """Output from the inference engine."""

    symbol: str
    timestamp: str
    prediction: float
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_version: str = ""
    feature_version: str = ""
    is_calibrated: bool = False
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureRuntimeResult:
    """
    Full pipeline result returned by FeatureRuntimeService.
    """

    status: FeatureRuntimeStatus
    symbol: str
    timestamp: str

    feature_snapshot: Optional[FeatureSnapshot] = None
    prediction_result: Optional[PredictionResult] = None
    ml_signal: Optional[MLSignal] = None

    failed_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
