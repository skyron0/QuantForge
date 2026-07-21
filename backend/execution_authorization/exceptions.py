class ExecutionAuthorizationError(Exception):
    """Base exception for all execution authorization errors."""
    pass


class InvalidExecutionPolicyError(ExecutionAuthorizationError):
    """Raised when the execution policy is invalid or improperly configured."""
    pass


class ExecutionValidationError(ExecutionAuthorizationError):
    """Raised when input validation on execution parameters fails."""
    pass


class LineageMismatchError(ExecutionAuthorizationError):
    """Raised when the lineage of trade components is inconsistent (e.g. proposal, risk, sizing mismatch)."""
    pass


class ExecutionDisabledError(ExecutionAuthorizationError):
    """Raised when global execution is disabled in the system context."""
    pass


class KillSwitchActiveError(ExecutionAuthorizationError):
    """Raised when the system kill-switch is active."""
    pass


class SymbolTradingDisabledError(ExecutionAuthorizationError):
    """Raised when trading for the specific symbol is disabled."""
    pass


class StaleMarketDataError(ExecutionAuthorizationError):
    """Raised when market data exceeds the freshness age threshold or is future-dated."""
    pass


class DuplicateIntentError(ExecutionAuthorizationError):
    """Raised when a duplicate execution intent is detected for the same logical trade."""
    pass


class InvalidOrderIntentError(ExecutionAuthorizationError):
    """Raised when an OrderIntent structure is invalid."""
    pass


class LiveExecutionNotAllowedError(ExecutionAuthorizationError):
    """Raised when live execution is attempted but not explicitly permitted by environment and policy."""
    pass
