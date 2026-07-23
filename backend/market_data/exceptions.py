class MarketDataError(Exception):
    """Base exception for all market data package errors."""
    pass


class MarketDataValidationError(MarketDataError):
    """Raised when market data fails structural or semantic validation."""
    pass


class InvalidMarketDataPolicyError(MarketDataError):
    """Raised when a market data policy is invalid or violated."""
    pass


class UnsupportedMarketDataTypeError(MarketDataValidationError):
    """Raised when market data type is not supported by the platform."""
    pass


class ProviderError(MarketDataError):
    """Raised for general market data provider failures."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unreachable or disconnected."""
    pass


class NormalizationError(MarketDataError):
    """Raised when raw feed normalization fails."""
    pass


class InvalidTimestampError(MarketDataValidationError):
    """Raised when timestamp is malformed, missing, or invalid."""
    pass


class StaleMarketDataError(MarketDataValidationError):
    """Raised when market data timestamp is older than allowed by policy."""
    pass


class FutureMarketDataError(MarketDataValidationError):
    """Raised when market data timestamp is too far in the future."""
    pass


class SequenceError(MarketDataError):
    """Base exception for sequence integrity issues."""
    pass


class SequenceGapError(SequenceError):
    """Raised when a gap in the message sequence is detected."""
    pass


class DuplicateMarketDataError(SequenceError):
    """Raised when a duplicate sequence message is received."""
    pass


class OutOfOrderMarketDataError(SequenceError):
    """Raised when a message is received out of sequential order."""
    pass


class MarketDataStoreError(MarketDataError):
    """Raised for errors in the in-memory market data storage buffers."""
    pass


class SnapshotError(MarketDataError):
    """Raised when a market data snapshot builder fails or cannot produce a valid snapshot."""
    pass
