class InferenceError(Exception):
    """Base exception for all real-time inference engine failures."""
    pass


class ModelNotFoundError(InferenceError):
    """Raised when a requested model version is not registered or found."""
    pass


class ModelLoadError(InferenceError):
    """Raised when a model artifact cannot be successfully loaded or is corrupted."""
    pass


class LifecycleError(InferenceError):
    """Raised when an ineligible model status is requested for runtime inference."""
    pass


class SchemaValidationError(InferenceError):
    """Raised when runtime feature schema does not match model expectations."""
    pass


class PredictionError(InferenceError):
    """Raised when prediction execution throws an error under the framework adapter."""
    pass


class RegistryError(InferenceError):
    """Raised when query or lookup operations in the ModelRegistry fail."""
    pass


class ArtifactIntegrityError(InferenceError):
    """Raised when SHA-256 validation of model or calibration artifacts fails."""
    pass
