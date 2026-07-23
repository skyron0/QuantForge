class RuntimeException(Exception):
    """Base exception for all trading runtime exceptions."""
    pass


class InvalidStateTransitionError(RuntimeException):
    """Raised when trying to perform an invalid state transition in TradingRuntime."""
    pass


class PolicyValidationError(RuntimeException):
    """Raised when validation of RuntimePolicy configurations fails."""
    pass


class PublishError(RuntimeException):
    """Raised when publishing an event to EventBus fails (e.g. queue overflow)."""
    pass


class DispatchError(RuntimeException):
    """Raised when dispatching an event encounters critical system/boundary errors."""
    pass


class SubscriberError(RuntimeException):
    """Raised when an event subscriber handler throws an error."""
    pass


class SessionError(RuntimeException):
    """Raised when TradingSession state integrity checks fail."""
    pass


class SchedulerError(RuntimeException):
    """Raised when Scheduler execution or state checks fail."""
    pass
