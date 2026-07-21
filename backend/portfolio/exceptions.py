class PortfolioError(Exception):
    """Base exception for all portfolio errors."""
    pass

class PortfolioValidationError(PortfolioError):
    """Raised when validating portfolio parameters/states fails."""
    pass

class InvalidPortfolioPolicyError(ValueError, PortfolioError):
    """Raised when portfolio policy configuration is invalid."""
    pass

class InvalidFillError(PortfolioError):
    """Raised when an ingested fill is invalid (e.g. invalid price/qty)."""
    pass

class DuplicateFillError(PortfolioError):
    """Raised when a fill identifier has already been processed."""
    pass

class PositionAccountingError(PortfolioError):
    """Raised when position mathematical updates fail or would violate boundaries."""
    pass

class InsufficientPositionError(PortfolioError):
    """Raised when attempting to close more position quantity than available."""
    pass

class InvalidPositionTransitionError(PortfolioError):
    """Raised when a position transition violates state rules."""
    pass

class PortfolioInvariantError(PortfolioError):
    """Raised when a portfolio invariant is violated (e.g., negative balance, mismatches)."""
    pass

class UnsupportedInstrumentError(PortfolioError):
    """Raised when an instrument type is not supported by the manager."""
    pass
