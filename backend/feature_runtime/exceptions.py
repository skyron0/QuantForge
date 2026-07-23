"""
Feature Runtime custom exception hierarchy.

All exceptions derive from FeatureRuntimeError to allow callers to catch
the entire category with a single handler.
"""


class FeatureRuntimeError(Exception):
    """Base exception for all Feature Runtime errors."""
    pass


# ─── Schema & Validation ────────────────────────────────────────────────────

class FeatureValidationError(FeatureRuntimeError):
    """Raised when a feature snapshot fails bounds or type validation."""
    pass


class InvalidFeaturePolicyError(FeatureRuntimeError):
    """Raised when the FeatureRuntimePolicy is invalid."""
    pass


class FeatureSchemaError(FeatureRuntimeError):
    """Raised for generic schema definition issues."""
    pass


class FeatureSchemaMismatchError(FeatureRuntimeError):
    """Raised when the computed schema fingerprint does not match expectations."""
    pass


class FeatureOrderingError(FeatureRuntimeError):
    """Raised when feature ordering does not match the canonical schema."""
    pass


# ─── History & Warmup ────────────────────────────────────────────────────────

class InsufficientHistoryError(FeatureRuntimeError):
    """Raised when the historical buffer lacks minimum candle count."""
    pass


class FeatureWarmupError(FeatureRuntimeError):
    """Raised when the system is still in warmup phase."""
    pass


# ─── Value Integrity ────────────────────────────────────────────────────────

class MissingFeatureError(FeatureRuntimeError):
    """Raised when expected feature keys are absent from the snapshot."""
    pass


class InvalidFeatureValueError(FeatureRuntimeError):
    """Raised when a feature value is NaN, Inf, or otherwise invalid."""
    pass


class StaleFeatureError(FeatureRuntimeError):
    """Raised when feature data exceeds the staleness time limit."""
    pass


class FutureFeatureError(FeatureRuntimeError):
    """Raised when a causal violation is detected (look-ahead leakage)."""
    pass


class UnsupportedFeatureError(FeatureRuntimeError):
    """Raised when an unknown or unsupported feature type is requested."""
    pass


# ─── Inference ───────────────────────────────────────────────────────────────

class InferenceRuntimeError(FeatureRuntimeError):
    """Raised on inference execution failures."""
    pass


class ModelUnavailableError(FeatureRuntimeError):
    """Raised when the required model cannot be loaded."""
    pass


class ModelCompatibilityError(FeatureRuntimeError):
    """Raised when model metadata does not match the feature schema."""
    pass


class InvalidPredictionError(FeatureRuntimeError):
    """Raised when the model produces an invalid prediction value."""
    pass


# ─── Signal ──────────────────────────────────────────────────────────────────

class SignalGenerationError(FeatureRuntimeError):
    """Raised when the prediction-to-MLSignal mapping fails."""
    pass
