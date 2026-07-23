class ApplicationRuntimeError(Exception):
    """Base exception for all application runtime errors."""
    pass


class ApplicationConfigurationError(ApplicationRuntimeError):
    """Raised when application configuration is invalid or missing."""
    pass


class ComponentInitializationError(ApplicationRuntimeError):
    """Raised when a component fails to initialize."""
    pass


class ComponentDependencyError(ApplicationRuntimeError):
    """Raised when a dependency check between components fails."""
    pass


class RuntimeCoordinationError(ApplicationRuntimeError):
    """Raised when coordination between engines fails."""
    pass


class TradingCycleCoordinationError(RuntimeCoordinationError):
    """Raised during errors in the trading cycle orchestration."""
    pass


class MarketDataCoordinationError(RuntimeCoordinationError):
    """Raised during market data pipeline coordination errors."""
    pass


class FeatureRuntimeCoordinationError(RuntimeCoordinationError):
    """Raised during feature runtime coordination errors."""
    pass


class PersistenceCoordinationError(RuntimeCoordinationError):
    """Raised during persistence service coordination errors."""
    pass


class GracefulShutdownError(ApplicationRuntimeError):
    """Raised when graceful shutdown sequence fails."""
    pass


class HealthCheckError(ApplicationRuntimeError):
    """Raised when health monitor checks fail or detect critical issues."""
    pass


class SessionInitializationError(ApplicationRuntimeError):
    """Raised when trading session initialization fails."""
    pass
