class IntelligenceError(Exception):
    """Base exception for all intelligence and reasoning errors."""
    pass


class ProviderError(IntelligenceError):
    """Base error for AI Provider failures."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when the AI Provider server is unreachable or offline."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when an AI Provider call times out (connection or inference)."""
    pass


class ProviderResponseError(ProviderError):
    """Raised when the AI Provider returns an invalid/unexpected API response code or structure."""
    pass


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is missing, invalid, or corrupted."""
    pass


class StructuredOutputError(IntelligenceError):
    """Raised when LLM output cannot be parsed into JSON or fails schema/domain validation."""
    pass


class ReasoningError(IntelligenceError):
    """Raised when high-level reasoning orchestrator fails to conclude."""
    pass
