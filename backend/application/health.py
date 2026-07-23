from datetime import datetime, timezone
from typing import Dict, Any, Optional
from backend.application.models import ComponentHealth, ComponentHealthStatus
from backend.application.exceptions import HealthCheckError
from backend.market_data.provider import BaseMarketDataProvider
from backend.market_data.service import MarketDataService
from backend.feature_runtime.service import FeatureRuntimeService
from backend.runtime.runtime import TradingRuntime
from backend.runtime.event_bus import BaseEventBus
from backend.execution_adapter.base import BaseExecutionAdapter
from backend.portfolio.portfolio import PortfolioEngine
from backend.persistence.service import PersistenceService


class ComponentHealthMonitor:
    """
    Sub-system health monitor checking the live state, connection status,
    and credentials of orchestrator and infrastructure components.
    Does NOT execute trades or perform mutating operations.
    """
    def __init__(
        self,
        market_data_provider: Optional[BaseMarketDataProvider] = None,
        market_data_service: Optional[MarketDataService] = None,
        feature_runtime_service: Optional[FeatureRuntimeService] = None,
        inference_provider: Optional[Any] = None,  # e.g., BaseAIProvider
        trading_runtime: Optional[TradingRuntime] = None,
        event_bus: Optional[BaseEventBus] = None,
        execution_adapter: Optional[BaseExecutionAdapter] = None,
        portfolio_engine: Optional[PortfolioEngine] = None,
        persistence_service: Optional[PersistenceService] = None
    ) -> None:
        self.market_data_provider = market_data_provider
        self.market_data_service = market_data_service
        self.feature_runtime_service = feature_runtime_service
        self.inference_provider = inference_provider
        self.trading_runtime = trading_runtime
        self.event_bus = event_bus
        self.execution_adapter = execution_adapter
        self.portfolio_engine = portfolio_engine
        self.persistence_service = persistence_service

    def get_component_health(self, component_name: str) -> ComponentHealth:
        """Runs a check on a single component by name, returning ComponentHealth."""
        now = datetime.now(timezone.utc).isoformat()
        
        if component_name == "market_data_provider":
            if not self.market_data_provider:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Market data provider not registered.",
                    checked_at=now
                )
            try:
                healthy = self.market_data_provider.is_healthy()
                status = ComponentHealthStatus.HEALTHY if healthy else ComponentHealthStatus.UNHEALTHY
                msg = self.market_data_provider.get_health_status()
                return ComponentHealth(
                    component_name=component_name,
                    status=status,
                    message=f"Provider state: {msg}",
                    checked_at=now,
                    metadata={"is_healthy": healthy}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Provider check failed: {str(e)}",
                    checked_at=now
                )

        elif component_name == "market_data_service":
            if not self.market_data_service:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Market data service not registered.",
                    checked_at=now
                )
            # Service is stateless but relies on sequence tracker and store
            try:
                # Basic sanity check
                store_size = len(self.market_data_service.store.get_symbols()) if self.market_data_service.store else 0
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.HEALTHY,
                    message="Market data service active.",
                    checked_at=now,
                    metadata={"store_size": store_size}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Service check failed: {str(e)}",
                    checked_at=now
                )

        elif component_name == "feature_runtime":
            if not self.feature_runtime_service:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Feature runtime service not registered.",
                    checked_at=now
                )
            try:
                # If feature service itself works, we can check basic policy/warmup limits
                policy_version = self.feature_runtime_service.policy.policy_version
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.HEALTHY,
                    message=f"Feature runtime active (policy: {policy_version}).",
                    checked_at=now,
                    metadata={"policy_version": policy_version}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Feature runtime check failed: {str(e)}",
                    checked_at=now
                )

        elif component_name == "inference":
            if not self.inference_provider:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.HEALTHY,
                    message="Optional intelligence context system not loaded.",
                    checked_at=now,
                    metadata={"optional": True}
                )
            try:
                h = self.inference_provider.health_check()
                status = ComponentHealthStatus.HEALTHY if h.status == "healthy" else ComponentHealthStatus.UNHEALTHY
                return ComponentHealth(
                    component_name=component_name,
                    status=status,
                    message=f"Model: {h.model_name}, status: {h.status}",
                    checked_at=now,
                    metadata={"error": h.error_message}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Inference health check failed: {str(e)}",
                    checked_at=now
                )

        elif component_name == "trading_runtime":
            if not self.trading_runtime:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Trading runtime not registered.",
                    checked_at=now
                )
            # Check state
            state = self.trading_runtime.state
            status = ComponentHealthStatus.HEALTHY
            if state in ("STOPPED", "FAILED"):
                status = ComponentHealthStatus.UNHEALTHY
            elif state == "PAUSED":
                status = ComponentHealthStatus.DEGRADED
            return ComponentHealth(
                component_name=component_name,
                status=status,
                message=f"Runtime is in {state} state.",
                checked_at=now,
                metadata={"state": state}
            )

        elif component_name == "event_bus":
            if not self.event_bus:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="EventBus not registered.",
                    checked_at=now
                )
            return ComponentHealth(
                component_name=component_name,
                status=ComponentHealthStatus.HEALTHY,
                message="EventBus active.",
                checked_at=now
            )

        elif component_name == "execution_adapter":
            if not self.execution_adapter:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Execution adapter not registered.",
                    checked_at=now
                )
            # Enforce paper only constraint
            try:
                policy_env = self.execution_adapter.policy.environment
                from backend.execution_authorization.models import ExecutionEnvironment
                if policy_env == ExecutionEnvironment.LIVE:
                    return ComponentHealth(
                        component_name=component_name,
                        status=ComponentHealthStatus.UNHEALTHY,
                        message="CRITICAL: Live trading adapter detected. Disallowed by safety limits.",
                        checked_at=now
                    )
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.HEALTHY,
                    message=f"Adapter active in environment: {policy_env.name}.",
                    checked_at=now,
                    metadata={"environment": policy_env.name}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Adapter check failed: {str(e)}",
                    checked_at=now
                )

        elif component_name == "portfolio_engine":
            if not self.portfolio_engine:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Portfolio engine not registered.",
                    checked_at=now
                )
            try:
                p_state = self.portfolio_engine.get_state()
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.HEALTHY,
                    message="Portfolio state loaded successfully.",
                    checked_at=now,
                    metadata={"equity": float(p_state.equity), "positions_count": p_state.open_position_count}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Portfolio state check failed: {str(e)}",
                    checked_at=now
                )

        elif component_name == "persistence":
            if not self.persistence_service:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message="Persistence service not registered.",
                    checked_at=now
                )
            # Memory falls back. If Postgres is used but unreachable, DEGRADED or UNHEALTHY
            try:
                enabled = self.persistence_service._is_enabled()
                if not enabled:
                    return ComponentHealth(
                        component_name=component_name,
                        status=ComponentHealthStatus.DEGRADED,
                        message="Persistence is disabled.",
                        checked_at=now,
                        metadata={"enabled": False}
                    )
                # Quick write/read dry-run/check logic or checking connection pools
                # We can check pool health or db backend
                backend = self.persistence_service.policy.backend
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.HEALTHY,
                    message=f"Persistence active with backend: {backend}.",
                    checked_at=now,
                    metadata={"backend": backend}
                )
            except Exception as e:
                return ComponentHealth(
                    component_name=component_name,
                    status=ComponentHealthStatus.UNHEALTHY,
                    message=f"Persistence check failed: {str(e)}",
                    checked_at=now
                )

        return ComponentHealth(
            component_name=component_name,
            status=ComponentHealthStatus.UNHEALTHY,
            message="Unknown component requested.",
            checked_at=now
        )

    def check_all(self) -> Dict[str, ComponentHealth]:
        """Runs health checks on all components and aggregates results."""
        names = [
            "market_data_provider", "market_data_service", "feature_runtime",
            "inference", "trading_runtime", "event_bus", "execution_adapter",
            "portfolio_engine", "persistence"
        ]
        return {name: self.get_component_health(name) for name in names}
