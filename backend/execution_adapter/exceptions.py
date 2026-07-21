class ExecutionAdapterError(Exception):
    """Base exception for all execution adapter errors."""
    pass

class ExecutionAdapterValidationError(ExecutionAdapterError):
    """Exception raised when input validation fails."""
    pass

class InvalidExecutionAdapterPolicyError(ExecutionAdapterError):
    """Exception raised when policy configuration is invalid."""
    pass

class UnsupportedExecutionEnvironmentError(ExecutionAdapterError):
    """Exception raised when environment is not supported."""
    pass

class UnsupportedOrderTypeError(ExecutionAdapterError):
    """Exception raised when order type is invalid/unsupported."""
    pass

class StaleExecutionContextError(ExecutionAdapterError):
    """Exception raised when market context timestamp is stale or skewed."""
    pass

class InsufficientLiquidityError(ExecutionAdapterError):
    """Exception raised when available liquidity is insufficient."""
    pass

class DuplicateExecutionError(ExecutionAdapterError):
    """Exception raised when attempting to execute an intent that is already executed/claimed."""
    pass

class InvalidMarketStateError(ExecutionAdapterError):
    """Exception raised when market price, bid, or ask values are invalid."""
    pass

class ExecutionSimulationError(ExecutionAdapterError):
    """Exception raised for general simulator failures."""
    pass
