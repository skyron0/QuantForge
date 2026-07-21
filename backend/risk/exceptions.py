class RiskError(Exception):
    """Base exception for all risk validation errors."""
    pass


class InvalidRiskPolicyError(RiskError):
    """Raised when a risk policy is validation-wise invalid."""
    pass


class RiskValidationError(RiskError):
    """Base class for risk gate validation failures."""
    pass


class ProposalExpiredError(RiskValidationError):
    """Raised when a TradeProposal has expired or is too old."""
    pass


class RiskContextError(RiskError):
    """Raised when there is an issue with the RiskContext inputs (e.g. malformed/NaN values)."""
    pass


class RiskAuthorizationError(RiskError):
    """Raised for general failures during the risk authorization pipeline."""
    pass
