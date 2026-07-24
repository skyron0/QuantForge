import logging
from typing import Optional, Any, Callable, List
from decimal import Decimal
from backend.replay.clock import Clock, SystemClock
from backend.application.exceptions import (
    ComponentInitializationError, ComponentDependencyError, ApplicationConfigurationError
)
from backend.application.policy import IntegratedRuntimePolicy
from backend.application.telemetry import (
    IntegratedRuntimeTelemetrySink, ConsoleIntegratedRuntimeTelemetrySink
)
from backend.application.health import ComponentHealthMonitor

# Infrastructure and Runtime
from backend.runtime.event_bus import EventBus
from backend.runtime.dispatcher import Dispatcher
from backend.runtime.session import TradingSession
from backend.runtime.runtime import TradingRuntime, RuntimePolicy
from backend.runtime.telemetry import RuntimeTelemetry

# Domain Engines
from backend.market_data.service import MarketDataService
from backend.market_data.provider import BaseMarketDataProvider
from backend.market_data.normalizer import MarketDataNormalizer
from backend.market_data.validator import MarketDataValidator
from backend.market_data.sequence import SequenceTracker
from backend.market_data.store import MarketDataStore
from backend.market_data.snapshot import MarketDataSnapshotBuilder
from backend.market_data.telemetry import ConsoleMarketDataTelemetrySink
from backend.market_data.bridge import MarketDataBridge
from backend.market_data.policy import MarketDataPolicy

from backend.feature_runtime.service import FeatureRuntimeService
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.buffer import HistoricalFeatureBuffer, BufferCandle
from backend.feature_runtime.extractor import FeatureExtractor
from backend.feature_runtime.validator import FeatureValidator
from backend.feature_runtime.inference import FeatureInferenceEngine
from backend.feature_runtime.signal import FeatureSignalMapper
from backend.feature_runtime.telemetry import ConsoleFeatureRuntimeTelemetrySink
from backend.feature_runtime.bridge import FeatureRuntimeBridge
from backend.feature_runtime.schema import FeatureSchema

from backend.decision.fusion import DecisionFusionEngine
from backend.decision.policy import FusionPolicy
from backend.decision.telemetry import ConsoleDecisionTelemetrySink
from backend.decision.intelligence_context import IntelligenceContextStore

from backend.risk.guard import RiskGuardEngine
from backend.risk.policy import RiskPolicy
from backend.risk.telemetry import ConsoleRiskTelemetrySink

from backend.positioning.sizing import PositionSizingEngine
from backend.positioning.policy import PositionSizingPolicy
from backend.positioning.telemetry import ConsolePositionSizingTelemetrySink

from backend.execution_authorization.authorization import ExecutionAuthorizationEngine
from backend.execution_authorization.idempotency import IdempotencyStore
from backend.position_lifecycle.bridge import ExitAuthorizationEngine
from backend.execution_authorization.policy import ExecutionPolicy
from backend.execution_authorization.telemetry import ConsoleExecutionAuthorizationTelemetrySink
from backend.execution_authorization.models import ExecutionEnvironment

from backend.execution_adapter.paper import PaperExecutionAdapter
from backend.execution_adapter.policy import PaperExecutionPolicy
from backend.execution_adapter.telemetry import ConsoleExecutionTelemetrySink

from backend.portfolio.portfolio import PortfolioEngine
from backend.portfolio.policy import PortfolioPolicy
from backend.portfolio.telemetry import ConsolePortfolioTelemetrySink
from backend.portfolio.idempotency import FillIdempotencyStore

from backend.position_lifecycle.lifecycle import PositionLifecycleEngine
from backend.position_lifecycle.policy import PositionLifecyclePolicy
from backend.position_lifecycle.telemetry import ConsolePositionLifecycleTelemetrySink
from backend.position_lifecycle.store import PositionLifecycleStore

from backend.orchestration.orchestrator import TradingCycleOrchestrator
from backend.orchestration.policy import TradingCyclePolicy
from backend.orchestration.telemetry import ConsoleTradingCycleTelemetrySink

from backend.persistence.service import PersistenceService
from backend.persistence.policy import PersistencePolicy
from backend.persistence.telemetry import PersistenceTelemetry
from backend.persistence.bridge import PersistenceEventHandler


class QuantForgeContainer:
    """
    Composition root responsible for constructing and wiring all domain engines,
    bridges, persistence handlers, and scheduler components cleanly.
    Enforces PAPER-ONLY execution.
    """
    def __init__(
        self,
        runtime_policy: IntegratedRuntimePolicy,
        market_data_policy: MarketDataPolicy,
        feature_policy: FeatureRuntimePolicy,
        feature_schema: FeatureSchema,
        predict_fn: Any,  # Callable[[List[float]], float]
        fusion_policy: FusionPolicy,
        risk_policy: RiskPolicy,
        sizing_policy: PositionSizingPolicy,
        execution_policy: ExecutionPolicy,
        paper_exec_policy: PaperExecutionPolicy,
        portfolio_policy: PortfolioPolicy,
        lifecycle_policy: PositionLifecyclePolicy,
        cycle_policy: TradingCyclePolicy,
        persistence_policy: PersistencePolicy,
        market_data_provider: BaseMarketDataProvider,
        session_id: Optional[str] = None,
        clock: Optional[Clock] = None,
        uuid_generator: Optional[Callable[[], str]] = None
    ) -> None:
        self.runtime_policy = runtime_policy
        self.market_data_policy = market_data_policy
        self.feature_policy = feature_policy
        self.feature_schema = feature_schema
        self.predict_fn = predict_fn
        self.fusion_policy = fusion_policy
        self.risk_policy = risk_policy
        self.sizing_policy = sizing_policy
        self.execution_policy = execution_policy
        self.paper_exec_policy = paper_exec_policy
        self.portfolio_policy = portfolio_policy
        self.lifecycle_policy = lifecycle_policy
        self.cycle_policy = cycle_policy
        self.persistence_policy = persistence_policy
        self.market_data_provider = market_data_provider
        self.session_id = session_id
        self.clock = clock or SystemClock()
        self.uuid_generator = uuid_generator

        # Logger
        self.logger = logging.getLogger("QuantForge.Container")

        # Strict Paper-Only Validation
        if (ExecutionEnvironment.LIVE in execution_policy.allowed_environments 
                or execution_policy.allow_live_execution_intents):
            raise ApplicationConfigurationError(
                "CRITICAL ERROR: QuantForgeContainer cannot be initialized with LIVE ExecutionEnvironment."
            )

        # Construct and wire everything
        self._build_components()
        self._wire_event_bus()

    def _build_components(self) -> None:
        try:
            # 1. Telemetry and Session
            self.app_telemetry = ConsoleIntegratedRuntimeTelemetrySink()
            self.session = TradingSession(
                session_id=self.session_id,
                metadata={"policy_version": self.runtime_policy.policy_version}
            )

            # 2. Event Dispatcher & Bus
            self.dispatcher = Dispatcher()
            self.event_bus = EventBus(dispatcher=self.dispatcher)

            # 3. Market Data
            self.market_normalizer = MarketDataNormalizer()
            self.market_validator = MarketDataValidator(policy=self.market_data_policy)
            self.sequence_tracker = SequenceTracker(policy=self.market_data_policy)
            self.market_store = MarketDataStore(policy=self.market_data_policy)
            self.market_telemetry = ConsoleMarketDataTelemetrySink()
            self.market_service = MarketDataService(
                normalizer=self.market_normalizer,
                validator=self.market_validator,
                sequence_tracker=self.sequence_tracker,
                store=self.market_store,
                telemetry=self.market_telemetry,
                policy=self.market_data_policy,
                clock=self.clock
            )
            self.market_snapshot_builder = MarketDataSnapshotBuilder(
                store=self.market_store,
                sequence_tracker=self.sequence_tracker,
                policy=self.market_data_policy
            )
            self.market_bridge = MarketDataBridge(
                event_bus=self.event_bus,
                market_data_service=self.market_service,
                snapshot_builder=self.market_snapshot_builder,
                runtime_id="app-runtime",
                session_id=self.session.session_id
            )

            # 4. Feature Runtime
            self.feature_buffer = HistoricalFeatureBuffer(
                capacity=self.feature_policy.minimum_history * 2
            )
            self.feature_extractor = FeatureExtractor(
                schema=self.feature_schema,
                buffer=self.feature_buffer,
                minimum_history=self.feature_policy.minimum_history
            )
            self.feature_validator = FeatureValidator(schema=self.feature_schema)
            self.feature_inference = FeatureInferenceEngine(
                schema=self.feature_schema,
                predict_fn=self.predict_fn
            )
            self.feature_mapper = FeatureSignalMapper(
                bullish_threshold=self.feature_policy.bullish_threshold,
                bearish_threshold=self.feature_policy.bearish_threshold,
                default_timeframe=self.feature_policy.default_timeframe
            )
            self.feature_telemetry = ConsoleFeatureRuntimeTelemetrySink()
            self.feature_service = FeatureRuntimeService(
                policy=self.feature_policy,
                schema=self.feature_schema,
                buffer=self.feature_buffer,
                extractor=self.feature_extractor,
                validator=self.feature_validator,
                inference_engine=self.feature_inference,
                signal_mapper=self.feature_mapper,
                telemetry_sink=self.feature_telemetry,
            )
            self.feature_bridge = FeatureRuntimeBridge(
                event_bus=self.event_bus,
                service=self.feature_service,
                symbols=self.runtime_policy.enabled_symbols,
                runtime_id="app-runtime",
                session_id=self.session.session_id
            )

            # 5. Intelligence context store
            self.intel_store = IntelligenceContextStore(
                default_ttl_seconds=3600.0,
                max_size=100
            )

            # 6. Core Domain Engines
            self.decision_telemetry = ConsoleDecisionTelemetrySink()
            self.decision_fusion_engine = DecisionFusionEngine(
                policy=self.fusion_policy,
                telemetry_sink=self.decision_telemetry
            )

            self.risk_telemetry = ConsoleRiskTelemetrySink()
            self.risk_guard_engine = RiskGuardEngine(
                policy=self.risk_policy,
                telemetry_sink=self.risk_telemetry
            )

            self.sizing_telemetry = ConsolePositionSizingTelemetrySink()
            self.position_sizing_engine = PositionSizingEngine(
                policy=self.sizing_policy,
                telemetry_sink=self.sizing_telemetry,
                clock=self.clock,
                uuid_generator=self.uuid_generator
            )

            self.exec_idempotency = IdempotencyStore()
            self.execution_auth_telemetry = ConsoleExecutionAuthorizationTelemetrySink()
            self.execution_authorization_engine = ExecutionAuthorizationEngine(
                policy=self.execution_policy,
                idempotency_store=self.exec_idempotency,
                telemetry_sink=self.execution_auth_telemetry,
                clock=self.clock,
                uuid_generator=self.uuid_generator
            )
            self.exit_authorization_engine = ExitAuthorizationEngine(
                self.execution_policy,
                clock=self.clock,
                uuid_generator=self.uuid_generator
            )

            self.execution_telemetry = ConsoleExecutionTelemetrySink()
            self.paper_execution_adapter = PaperExecutionAdapter(
                policy=self.paper_exec_policy,
                telemetry_sink=self.execution_telemetry,
                clock=self.clock,
                uuid_generator=self.uuid_generator
            )

            self.portfolio_idempotency = FillIdempotencyStore()
            self.portfolio_telemetry = ConsolePortfolioTelemetrySink()
            self.portfolio_engine = PortfolioEngine(
                portfolio_id="app-paper-portfolio",
                initial_balance=Decimal("100000.0"),
                policy=self.portfolio_policy,
                idempotency_store=self.portfolio_idempotency,
                telemetry_sink=self.portfolio_telemetry,
                clock=self.clock
            )

            self.lifecycle_store = PositionLifecycleStore()
            self.lifecycle_telemetry = ConsolePositionLifecycleTelemetrySink()
            self.position_lifecycle_engine = PositionLifecycleEngine(
                policy=self.lifecycle_policy,
                store=self.lifecycle_store,
                telemetry_sink=self.lifecycle_telemetry
            )

            # 7. Cycle Orchestrator
            self.cycle_telemetry = ConsoleTradingCycleTelemetrySink()
            self.orchestrator = TradingCycleOrchestrator(
                policy=self.cycle_policy,
                decision_fusion_engine=self.decision_fusion_engine,
                risk_guard_engine=self.risk_guard_engine,
                position_sizing_engine=self.position_sizing_engine,
                execution_authorization_engine=self.execution_authorization_engine,
                exit_authorization_engine=self.exit_authorization_engine,
                paper_execution_adapter=self.paper_execution_adapter,
                portfolio_engine=self.portfolio_engine,
                position_lifecycle_engine=self.position_lifecycle_engine,
                telemetry_sink=self.cycle_telemetry,
                clock=self.clock,
                uuid_generator=self.uuid_generator
            )

            # 8. Trading Runtime (polling scheduler wrapper if used)
            self.runtime_telemetry = RuntimeTelemetry()
            self.trading_run_policy = RuntimePolicy(
                scheduler_interval_seconds=1.0
            )
            self.trading_runtime = TradingRuntime(
                policy=self.trading_run_policy,
                event_bus=self.event_bus,
                telemetry=self.runtime_telemetry,
                session=self.session,
                orchestrator=self.orchestrator
            )

            # 9. Persistence
            self.persistence_telemetry = PersistenceTelemetry(enabled=self.persistence_policy.persistence_enabled)
            self.persistence_service = PersistenceService(
                policy=self.persistence_policy,
                telemetry=self.persistence_telemetry
            )
            self.persistence_handler = PersistenceEventHandler(
                persistence_service=self.persistence_service
            )

            # 10. Health Monitor
            self.health_monitor = ComponentHealthMonitor(
                market_data_provider=self.market_data_provider,
                market_data_service=self.market_service,
                feature_runtime_service=self.feature_service,
                inference_provider=None,
                trading_runtime=self.trading_runtime,
                event_bus=self.event_bus,
                execution_adapter=self.paper_execution_adapter,
                portfolio_engine=self.portfolio_engine,
                persistence_service=self.persistence_service
            )

        except Exception as e:
            self.logger.error(f"Failed to build components: {str(e)}")
            raise ComponentInitializationError(
                f"Container build phase failed: {str(e)}"
            ) from e

    def _wire_event_bus(self) -> None:
        """Subscribes and links bridges and persistence handlers to the event stream."""
        try:
            # Wire persistence events
            self.persistence_handler.register(self.event_bus)
            
            # Subscribed components (e.g. bridging MarketDataSnapshot to FeatureRuntime)
            self.event_bus.subscribe(
                "MarketDataSnapshotUpdated",
                self._handle_snapshot_updates
            )
        except Exception as e:
            raise ComponentDependencyError(
                f"Failed to wire event bus subscriptions: {str(e)}"
            ) from e

    def _handle_snapshot_updates(self, event: Any) -> None:
        snap = getattr(event, "snapshot", None)
        if snap:
            # Send snapshot to Feature Runtime buffer for matching timeframe
            target_tf = self.feature_policy.default_timeframe
            candles = snap.candles.get(target_tf)
            if candles:
                c = candles[-1]
                self.feature_buffer.append(
                    snap.symbol,
                    BufferCandle(
                        timestamp=c.close_time,
                        open=float(c.open),
                        high=float(c.high),
                        low=float(c.low),
                        close=float(c.close),
                        volume=float(c.volume)
                    )
                )
            
            # Bridge to feature runtime processing
            self.feature_bridge.on_snapshot(snap.symbol, snap.timestamp)

    def check_dependencies(self) -> None:
        """Validates all components are non-None and healthy."""
        attribs = [
            "event_bus", "dispatcher", "session", "market_service",
            "market_snapshot_builder", "feature_service", "orchestrator",
            "trading_runtime", "persistence_service", "persistence_handler"
        ]
        for attr in attribs:
            if getattr(self, attr, None) is None:
                raise ComponentDependencyError(f"Required component: {attr} is not initialized.")
