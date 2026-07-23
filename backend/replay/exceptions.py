"""
Exceptions hierarchy for the historical replay system.
"""

class ReplayError(Exception):
    """Base exception for all historical replay errors."""
    pass

class ReplayValidationError(ReplayError):
    """Raised when replay configurations, policies, or inputs are invalid."""
    pass

class DatasetLoadingError(ReplayError):
    """Raised when loading or parsing seed datasets failed."""
    pass

class ReplayInvariantError(ReplayError):
    """Raised when safety, determinism, or causal constraints are breached during simulation."""
    pass
