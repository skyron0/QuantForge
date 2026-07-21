from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class InferenceRequest:
    """
    Standardized payload for real-time model inference requests.
    """
    model_version: str
    symbol: str
    timeframe: str
    features: Dict[str, float]
    timestamp: str  # ISO-8601 string
    feature_version: Optional[str] = None


@dataclass(frozen=True)
class InferenceMetadata:
    """
    Metadata associated with a successful inference run.
    """
    cache_hit: bool
    is_calibrated: bool
    confidence_source: str  # CALIBRATED, UNCALIBRATED, UNAVAILABLE
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResponse:
    """
    Standardized result returned on successful machine learning model predictions.
    Ineligible/failed requests raise the corresponding InferenceError subclass.
    """
    model_version: str
    symbol: str
    prediction: float  # Predicted class (for classification) or target value (regr)
    timestamp: str                               # Output timestamp
    latency_ms: float                            # Latency in milliseconds
    feature_version: str
    metadata: InferenceMetadata
    probabilities: Optional[List[float]] = None  # Class probabilities if classification
    confidence: Optional[float] = None          # Derived from probabilities if available
