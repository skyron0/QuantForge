"""
Feature Runtime & Online Inference Foundation (Sprint 3.11).

Deterministic, provider-independent bridge for online inference and ML signal
generation.  Strict architectural isolation from execution, risk, and external
AI API calls.  All operations fail closed on stale or invalid data.
"""

from backend.feature_runtime.exceptions import (
    FeatureRuntimeError,
    FeatureValidationError,
    FeatureSchemaMismatchError,
    FeatureOrderingError,
    InsufficientHistoryError,
    FeatureWarmupError,
    StaleFeatureError,
    FutureFeatureError,
    InferenceRuntimeError,
    ModelCompatibilityError,
    InvalidPredictionError,
    SignalGenerationError,
)
from backend.feature_runtime.models import (
    FeatureSnapshot,
    PredictionResult,
    FeatureRuntimeResult,
    FeatureRuntimeStatus,
)
from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.buffer import HistoricalFeatureBuffer
from backend.feature_runtime.extractor import FeatureExtractor
from backend.feature_runtime.validator import FeatureValidator
from backend.feature_runtime.inference import FeatureInferenceEngine
from backend.feature_runtime.signal import FeatureSignalMapper
from backend.feature_runtime.service import FeatureRuntimeService
from backend.feature_runtime.bridge import FeatureRuntimeBridge
from backend.feature_runtime.telemetry import (
    FeatureRuntimeTelemetrySink,
    ConsoleFeatureRuntimeTelemetrySink,
)

__all__ = [
    "FeatureRuntimeError",
    "FeatureValidationError",
    "FeatureSchemaMismatchError",
    "FeatureOrderingError",
    "InsufficientHistoryError",
    "FeatureWarmupError",
    "StaleFeatureError",
    "FutureFeatureError",
    "InferenceRuntimeError",
    "ModelCompatibilityError",
    "InvalidPredictionError",
    "SignalGenerationError",
    "FeatureSnapshot",
    "PredictionResult",
    "FeatureRuntimeResult",
    "FeatureRuntimeStatus",
    "FeatureSchema",
    "FeatureRuntimePolicy",
    "HistoricalFeatureBuffer",
    "FeatureExtractor",
    "FeatureValidator",
    "FeatureInferenceEngine",
    "FeatureSignalMapper",
    "FeatureRuntimeService",
    "FeatureRuntimeBridge",
    "FeatureRuntimeTelemetrySink",
    "ConsoleFeatureRuntimeTelemetrySink",
]
