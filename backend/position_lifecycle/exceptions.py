class PositionLifecycleError(Exception):
    """Base exception for all position lifecycle errors."""
    pass


class PositionLifecycleValidationError(PositionLifecycleError):
    """Raised when validation of position inputs or configurations fails."""
    pass


class InvalidLifecyclePolicyError(PositionLifecycleValidationError):
    """Raised when a lifecycle policy contains invalid settings or values."""
    pass


class ProtectiveLevelError(PositionLifecycleValidationError):
    """Base exception for invalid protective price level settings (e.g. SL or TP)."""
    pass


class InvalidStopLossError(ProtectiveLevelError):
    """Raised when stop loss settings are mathematically invalid or inconsistent."""
    pass


class InvalidTakeProfitError(ProtectiveLevelError):
    """Raised when take profit settings are mathematically invalid or inconsistent."""
    pass


class InvalidTrailingStopError(ProtectiveLevelError):
    """Raised when trailing stop configuration is invalid."""
    pass


class PositionNotFoundError(PositionLifecycleError):
    """Raised when an operation is requested on a position not present in the store."""
    pass


class PositionStateError(PositionLifecycleError):
    """Raised when an action is invalid for the current position or lifecycle state."""
    pass


class DuplicateTriggerError(PositionLifecycleError):
    """Raised when a trigger is evaluated for a lifecycle already in CLOSING/CLOSED status."""
    pass


class StaleMarketDataError(PositionLifecycleValidationError):
    """Raised when market prices or updates violate chronological freshness constraints."""
    pass


class LifecycleInvariantError(PositionLifecycleError):
    """Raised when internal state invariants are violated."""
    pass
