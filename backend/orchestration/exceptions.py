class OrchestrationError(Exception):
    """Base exception for all orchestration errors."""
    pass


class OrchestrationValidationError(OrchestrationError):
    """Raised when validation of inputs or context configuration fails."""
    pass


class StageExecutionError(OrchestrationError):
    """Raised when a specific domain engine fails during execution."""
    pass


class LineageIntegrityError(OrchestrationError):
    """Raised when there is a lineage correlation mismatch between engines."""
    pass


class PortfolioUpdateError(OrchestrationError):
    """Raised when updating the portfolio state fails."""
    pass


class LifecycleRegistrationError(OrchestrationError):
    """Raised when registering or updating protective state in the lifecycle store fails."""
    pass
