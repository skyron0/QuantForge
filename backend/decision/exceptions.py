class DecisionError(Exception):
    """Base exception for all decision-related errors."""
    pass


class InvalidPolicyError(DecisionError):
    """Raised when a FusionPolicy is invalid."""
    pass


class ContextStoreError(DecisionError):
    """Raised when an error occurs within the IntelligenceContextStore."""
    pass
