class PositionSizingError(Exception):
    """Base exception for all positioning-related errors."""
    pass


class InvalidPositionSizingPolicyError(PositionSizingError):
    """Raised when the positioning policy configuration is invalid."""
    pass


class PositionSizingValidationError(PositionSizingError):
    """Raised when sizing inputs or state validation fails."""
    pass


class InvalidStopDistanceError(PositionSizingValidationError):
    """Raised when stop distance is invalid, e.g., zero or mathematically incorrect orientation."""
    pass


class InsufficientCapitalError(PositionSizingValidationError):
    """Raised when capital or available balance/margin is insufficient."""
    pass


class PositionLimitError(PositionSizingValidationError):
    """Raised when final sized position exceeds policy Limits."""
    pass


class AuthorizationError(PositionSizingValidationError):
    """Raised when RiskAuthorizationResult is missing, rejected, stale, or incompatible."""
    pass
