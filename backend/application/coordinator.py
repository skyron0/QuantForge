import logging
import time
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional

from backend.application.exceptions import (
    ApplicationRuntimeError,
    ApplicationConfigurationError,
    ComponentInitializationError,
    ComponentDependencyError,
    HealthCheckError,
    SessionInitializationError,
)
from backend.application.container import QuantForgeContainer
from backend.application.models import (
    IntegratedRuntimeStatus,
    CycleTriggerType,
    IntegratedRuntimeSnapshot,
    SessionSummary,
)
from backend.application.policy import IntegratedRuntimePolicy

# Domain models
from backend.decision.models import MLSignal, IntelligenceSnapshot
from backend.risk.models import RiskContext, RiskAuthorizationStatus
from backend.positioning.models import PositionSizingContext
from backend.execution_authorization.models import ExecutionContext, ExecutionEnvironment
from backend.execution_adapter.models import PaperExecutionContext
from backend.orchestration.models import TradingCycleInput, TradingCycleResult, TradingCycleStatus
from backend.portfolio.models import PortfolioState
from backend.portfolio.bridge import PortfolioRiskContextBuilder
from backend.feature_runtime.models import FeatureRuntimeStatus


class IntegratedPaperTradingCoordinator:
    """
    Authoritative coordinator orchestrating continuous paper-only trading loops,
    driving session state transitions, validating component health,
    and routing protective exit vs. entry cycle flows.
    """
    def __init__(self, container: QuantForgeContainer) -> None:
        self.container = container
        self.policy: IntegratedRuntimePolicy = container.runtime_policy
        self.logger = logging.getLogger("QuantForge.Coordinator")

        # Synchronisation
        self._lock = threading.Lock()
        self._cycle_lock = threading.Lock()

        # Session tracking state
        self._status = IntegratedRuntimeStatus.INITIALIZED
        self._cycle_count = 0
        self._successful_cycles = 0
        self._rejected_cycles = 0
        self._failed_cycles = 0
        self._warmup_cycles = 0
        
        # Idempotency and duplicate prevention: mapping symbol -> last processed candle close or timestamp
        self._idempotency_map: Dict[str, str] = {}
        self._consecutive_failures = 0
        
        # Timing
        self._session_start_time: Optional[float] = None
        self._session_stop_time: Optional[float] = None

        # Verify container components
        self.container.check_dependencies()

    @property
    def status(self) -> IntegratedRuntimeStatus:
        with self._lock:
            return self._status

    @property
    def cycle_count(self) -> int:
        with self._lock:
            return self._cycle_count

    def start_session(self) -> None:
        """Starts the paper trading session: performs health check and updates status."""
        with self._lock:
            if self._status != IntegratedRuntimeStatus.INITIALIZED:
                raise SessionInitializationError(
                    f"Cannot start session. Expected status INITIALIZED, found {self._status.value}"
                )
            
            self.logger.info("Initializing integrated paper trading coordinator...")
            self._status = IntegratedRuntimeStatus.STARTING
            self._session_start_time = time.perf_counter()
            self.container.app_telemetry.record_session_start()

        # Run health checks to guarantee components are ready
        try:
            # Start market data provider
            self.container.market_data_provider.start()
            
            self._run_pre_flight_health_check()
            
            with self._lock:
                self._status = IntegratedRuntimeStatus.WARMING_UP
                self.logger.info("Session started. Warming up...")
        except Exception as e:
            with self._lock:
                self._status = IntegratedRuntimeStatus.FAILED
            self.stop_session("Pre-flight health check failure")
            raise SessionInitializationError(f"Pre-flight health check/startup failed: {str(e)}") from e

    def pause(self) -> None:
        """Pauses the runtime loop."""
        with self._lock:
            if self._status not in (IntegratedRuntimeStatus.WARMING_UP, IntegratedRuntimeStatus.RUNNING):
                self.logger.warning(f"Pause ignored. Session is in {self._status.value} state.")
                return
            self._status = IntegratedRuntimeStatus.PAUSED
            self.logger.info("Session paused.")

    def resume(self) -> None:
        """Resumes the runtime loop, restoring running status."""
        with self._lock:
            if self._status != IntegratedRuntimeStatus.PAUSED:
                self.logger.warning(f"Resume ignored. Session is in {self._status.value} state.")
                return
            
            # Decide if still warming up or running
            history_needed = self.container.feature_policy.minimum_history
            
            # We can check how many candles are in buffer
            # (Just count candles in historicalbuffer)
            # Find any symbol's buffer size
            symbols = self.policy.enabled_symbols
            warmed_up = True
            for sym in symbols:
                c_list = self.container.feature_buffer.get_candles_up_to(sym, "9999-12-31")
                if len(c_list) < history_needed:
                    warmed_up = False
                    break
                    
            self._status = IntegratedRuntimeStatus.RUNNING if warmed_up else IntegratedRuntimeStatus.WARMING_UP
            self.logger.info(f"Session resumed. Transformed to status: {self._status.value}")

    def stop_session(self, reason: str = "Manual stop") -> SessionSummary:
        """Stops the paper trading session: performs tear-down and returns SessionSummary."""
        with self._lock:
            if self._status in (IntegratedRuntimeStatus.STOPPED, IntegratedRuntimeStatus.STOPPING):
                # Return double-stopped summary
                return self._create_session_summary(reason)
                
            self._status = IntegratedRuntimeStatus.STOPPING
            self.logger.info(f"Stopping session. Reason: {reason}...")
            self._session_stop_time = time.perf_counter()

        # Tear down dependencies
        try:
            self.container.market_data_provider.stop()
        except Exception as e:
            self.logger.error(f"Error stopping market data provider: {str(e)}")

        with self._lock:
            self._status = IntegratedRuntimeStatus.STOPPED
            
        latency_ms = 0.0
        if self._session_start_time and self._session_stop_time:
            latency_ms = (self._session_stop_time - self._session_start_time) * 1000.0

        if reason == "Pre-flight health check failure":
            self.container.app_telemetry.record_session_failed(latency_ms)
        else:
            self.container.app_telemetry.record_session_complete(latency_ms)

        summary = self._create_session_summary(reason)
        self.logger.info(f"Session stopped. Realized PnL: {summary.realized_pnl:.2f}")
        return summary

    def step(
        self,
        trigger_type: CycleTriggerType,
        manual_signal: Optional[MLSignal] = None,
        manual_timestamp: Optional[str] = None
    ) -> List[TradingCycleResult]:
        """
        Executes a single workflow cycle deterministically across configured symbols.
        Checks protective exits first, then fires entry cycle checks.
        """
        # Enforce execution status checks
        current_status = self.status
        if current_status not in (
            IntegratedRuntimeStatus.WARMING_UP,
            IntegratedRuntimeStatus.RUNNING,
            IntegratedRuntimeStatus.STARTING
        ):
            self.logger.warning(f"Cycle step ignored. Session is in {current_status.value} state.")
            return []

        # Prevent duplicate concurrent cycles (sequential processing only)
        if not self._cycle_lock.acquire(blocking=False):
            self.logger.warning("Cycle step ignored. Another cycle is already executing.")
            return []

        try:
            results: List[TradingCycleResult] = []
            
            # Step timestamp
            now_iso = manual_timestamp or datetime.now(timezone.utc).isoformat()
            
            # 1. Component Health Checking (if needed)
            if self._cycle_count % self.policy.health_check_interval_cycles == 0:
                self._verify_operational_health()

            # 2. Check protective position exits (First step of cycle for all symbols)
            self._evaluate_protective_exits(now_iso)

            # 3. Process entry cycles symbol-by-symbol (sequential processing)
            for raw_sym in self.policy.enabled_symbols:
                # Normalize symbol at the application boundary
                symbol = raw_sym.replace("/", "").upper()
                
                cycle_result = self._process_symbol_entry(
                    symbol=symbol,
                    trigger_type=trigger_type,
                    now_iso=now_iso,
                    manual_signal=manual_signal
                )
                if cycle_result:
                    results.append(cycle_result)
                    
            return results

        finally:
            self._cycle_lock.release()

    def _process_symbol_entry(
        self,
        symbol: str,
        trigger_type: CycleTriggerType,
        now_iso: str,
        manual_signal: Optional[MLSignal] = None
    ) -> Optional[TradingCycleResult]:
        """Processes a single symbol's decision fusion, risk validation, and entry sizing/order execution."""
        # 1. Check/Build Market Data Snapshot
        try:
            # We bypass snapshot building if we are passing a manual signal in manual/replay test mode
            if manual_signal is not None:
                # Mock or build simple snapshot to get current ticker price
                ticker = self.container.market_store.get_ticker(symbol)
                if not ticker:
                    # Fabricate bare ticker for manual tests
                    from backend.market_data.models import TickerSnapshot
                    ticker = TickerSnapshot(
                        symbol=symbol,
                        bid=Decimal("100"),
                        ask=Decimal("100.1"),
                        last=Decimal("100.05"),
                        bid_quantity=Decimal("10"),
                        ask_quantity=Decimal("10"),
                        volume_24h=Decimal("1000"),
                        timestamp=now_iso,
                        source="manual",
                        received_at=now_iso
                    )
                    self.container.market_store.update_ticker(ticker)
            
            snapshot = self.container.market_snapshot_builder.build_snapshot(symbol, now_iso)
        except Exception as e:
            self.logger.warning(f"Skipping cycle for {symbol}: unable to construct market snapshot: {str(e)}")
            return None

        # 2. Fetch or trigger feature extraction
        # Handle warmup state detection
        history_needed = self.container.feature_policy.minimum_history
        timeframe = self.policy.enabled_timeframes[0]  # Take first default timeframe
        
        # Verify idempotency (prevent duplicates for same symbol + timeframe + candle close timestamp)
        if trigger_type == CycleTriggerType.CANDLE_CLOSE:
            # Get last closed candle
            candles = snapshot.candles.get(timeframe, [])
            if not candles:
                return None
            last_candle = candles[-1]
            if not last_candle.closed:
                # Wait for closed candle
                return None
            key = f"{symbol}:{timeframe}:{last_candle.close_time}"
            if self._idempotency_map.get(symbol) == key:
                # Already processed
                return None
            self._idempotency_map[symbol] = key
        else:
            # Replay step or manual - use timestamp idempotency
            key = f"{symbol}:{timeframe}:{now_iso}"
            if self._idempotency_map.get(symbol) == key:
                return None
            self._idempotency_map[symbol] = key

        # Increment cycle count
        with self._lock:
            self._cycle_count += 1
        self.container.app_telemetry.record_cycle_start()

        # Run feature processing
        feature_start = time.perf_counter()
        try:
            # Inject candle to buffer if not using bridge
            # Run extraction
            feat_result = self.container.feature_service.process(symbol, now_iso)
        except Exception as e:
            self.logger.error(f"Feature runtime evaluation failed for {symbol}: {str(e)}")
            self._handle_cycle_failure()
            return None

        # Check for warm-up state
        if feat_result.status == FeatureRuntimeStatus.WARMUP_SKIP:
            with self._lock:
                self._warmup_cycles += 1
                self._status = IntegratedRuntimeStatus.WARMING_UP
            self.container.app_telemetry.record_cycle_warmup()
            self.logger.info(f"Feature runtime warming up for {symbol}. Skipping downstream engines.")
            return None
        
        # Once out of warmup, automatically transition to running
        with self._lock:
            if self._status == IntegratedRuntimeStatus.WARMING_UP:
                self._status = IntegratedRuntimeStatus.RUNNING
                self.logger.info("Warmup complete. Session transitioned to RUNNING status.")

        # Fail closed for structural feature failures
        if feat_result.status != FeatureRuntimeStatus.SUCCESS and manual_signal is None:
            self.logger.warning(f"Feature pipeline returned unsuccessful status: {feat_result.status.value}")
            self._handle_cycle_failure()
            return None

        # Derive ML Signal
        ml_signal = manual_signal or feat_result.ml_signal
        if not ml_signal:
            # Warmup or neutral signal that outputs nothing
            return None
            
        self.container.app_telemetry.record_signal_generated()

        # 3. Decision Fusion & Intelligence Route (No blocking Ollama/cloud calls allowed)
        # Pull asynchronous intelligence snapshot if present
        intel_snap: Optional[IntelligenceSnapshot] = None
        timeframe = self.policy.enabled_timeframes[0]
        intel_snap = self.container.intel_store.get(symbol, timeframe)

        # 4. Construct contexts from current portfolio/accounting state dynamically (feedback loop)
        try:
            portfolio_state = self.container.portfolio_engine.get_state()
            
            # Derive current RiskContext
            risk_context = PortfolioRiskContextBuilder.build_risk_context(
                state=portfolio_state,
                symbol=symbol,
                volatility_state="NORMAL",
                market_liquidity_state="NORMAL"
            )
            
            # Stop loss calculation (uses default 2% or read parameters)
            ticker = snapshot.ticker
            entry_price = float(ticker.last)
            
            # Simple stop-loss percentage logic (2%)
            stop_loss_pct = 0.02
            if ml_signal.direction.upper() == "BULLISH":
                stop_loss_price = entry_price * (1.0 - stop_loss_pct)
            else:
                stop_loss_price = entry_price * (1.0 + stop_loss_pct)

            # Derive PositionSizingContext
            pos_sizing_context = PositionSizingContext(
                symbol=symbol,
                instrument_type="spot",
                equity=float(portfolio_state.equity),
                available_balance=float(portfolio_state.available_balance),
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                market_price=entry_price,
                leverage=1.0,
                contract_size=1.0,
                lot_size=0.001,
                min_quantity=0.001,
                max_quantity=1000.0,
                quantity_step=0.001,
                price_tick=0.01,
                current_symbol_exposure=float(pos.position_notional) if (pos := portfolio_state.positions.get(symbol)) is not None else 0.0,
                current_portfolio_exposure=float(portfolio_state.gross_exposure),
                market_timestamp=snapshot.timestamp,
                timestamp=now_iso
            )

            # Derive ExecutionContext
            exec_context = ExecutionContext(
                environment=ExecutionEnvironment.PAPER,
                current_timestamp=now_iso,
                market_timestamp=snapshot.timestamp,
                execution_enabled=True,
                kill_switch_active=False,
                symbol_trading_enabled=True,
                available_balance=float(portfolio_state.available_balance),
                current_price=entry_price
            )

            # Derive PaperExecutionContext
            paper_exec_context = PaperExecutionContext(
                current_market_price=entry_price,
                bid_price=float(ticker.bid),
                ask_price=float(ticker.ask),
                available_liquidity=1000000.0,
                timestamp=snapshot.timestamp
            )

        except Exception as e:
            self.logger.error(f"Failed to build session contexts for {symbol}: {str(e)}")
            self._handle_cycle_failure()
            return None

        # 5. Run the authoritative TradingCycleOrchestrator
        try:
            cycle_input = TradingCycleInput(
                ml_signal=ml_signal,
                risk_context=risk_context,
                position_sizing_context=pos_sizing_context,
                execution_context=exec_context,
                paper_execution_context=paper_exec_context,
                timestamp=now_iso,
                intelligence_snapshot=intel_snap
            )
            
            result = self.container.orchestrator.run_cycle(cycle_input)
            
            # Map result back to telemetry
            self._update_cycle_telemetry(result)
            return result

        except Exception as e:
            self.logger.error(f"Trading loop error in orchestrator: {str(e)}")
            self._handle_cycle_failure()
            return None

    def _evaluate_protective_exits(self, now_iso: str) -> None:
        """Evaluates and processes protective exits through the existing exit authorization chain."""
        try:
            portfolio_state = self.container.portfolio_engine.get_state()
            active_symbols = list(portfolio_state.positions.keys())
            
            for symbol in active_symbols:
                position = portfolio_state.positions[symbol]
                if float(position.quantity) == 0.0:
                    continue
                    
                position_id = position.position_id
                
                # Fetch ticker snapshot
                ticker = self.container.market_store.get_ticker(symbol)
                if not ticker:
                    continue
                    
                market_price = float(ticker.last)
                market_timestamp = ticker.timestamp

                # Check if position registered in lifecycle store. If not, auto-register to track protective exits
                lifecycle_state = self.container.position_lifecycle_engine.store.get(position_id)
                if not lifecycle_state:
                    side_lh = position.side.name
                    # Import enums
                    from backend.position_lifecycle.models import PositionSide
                    l_side = PositionSide.LONG if side_lh == "LONG" else PositionSide.SHORT
                    
                    stop_loss_val = position.metadata.get("stop_loss")
                    take_profit_val = position.metadata.get("take_profit")
                    self.container.position_lifecycle_engine.register_position(
                        position_id=position_id,
                        symbol=symbol,
                        side=l_side,
                        quantity=position.quantity,
                        average_entry_price=position.average_entry_price,
                        stop_loss=Decimal(str(stop_loss_val)) if stop_loss_val is not None else None,
                        take_profit=Decimal(str(take_profit_val)) if take_profit_val is not None else None,
                        trailing_stop_enabled=False,
                        trailing_distance=None,
                        trailing_activation_price=None,
                        timestamp=now_iso
                    )

                # Derive context mappings
                exec_context = ExecutionContext(
                    environment=ExecutionEnvironment.PAPER,
                    current_timestamp=now_iso,
                    market_timestamp=market_timestamp,
                    execution_enabled=True,
                    kill_switch_active=False,
                    symbol_trading_enabled=True,
                    available_balance=float(portfolio_state.available_balance),
                    current_price=market_price
                )
                
                paper_exec_context = PaperExecutionContext(
                    current_market_price=market_price,
                    bid_price=float(ticker.bid),
                    ask_price=float(ticker.ask),
                    available_liquidity=1000000.0,
                    timestamp=market_timestamp
                )

                idempotency_key = f"exit-{position_id}-{market_timestamp}"

                # Run exit check inside TradingCycleOrchestrator
                exit_result = self.container.orchestrator.run_exit_cycle(
                    position_id=position_id,
                    market_price=market_price,
                    market_timestamp=market_timestamp,
                    system_timestamp=now_iso,
                    execution_context=exec_context,
                    paper_execution_context=paper_exec_context,
                    idempotency_key=idempotency_key
                )
                
                if exit_result:
                    self.container.app_telemetry.record_protective_exit()
                    self._update_cycle_telemetry(exit_result)
                    
        except Exception as e:
            self.logger.error(f"Error evaluating protective exits: {str(e)}")

    def _update_cycle_telemetry(self, result: TradingCycleResult) -> None:
        """Translates orchestrator cycle results to coordinator-level counters and telemetry sinks."""
        with self._lock:
            # Keep counts
            if result.status == TradingCycleStatus.COMPLETED:
                self._successful_cycles += 1
                self._consecutive_failures = 0
            elif result.status in (
                TradingCycleStatus.NO_PROPOSAL,
                TradingCycleStatus.FUSION_REJECTED,
                TradingCycleStatus.RISK_REJECTED,
                TradingCycleStatus.SIZING_REJECTED,
                TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
            ):
                self._rejected_cycles += 1
                self._consecutive_failures = 0
            else:
                self._failed_cycles += 1
                self._consecutive_failures += 1

        self.container.app_telemetry.record_cycle_complete(result.latency_ms)

        # Map details to logger telemetry
        if result.status == TradingCycleStatus.COMPLETED:
            self.container.app_telemetry.record_position_opened()
        if result.status == TradingCycleStatus.RISK_REJECTED:
            self.container.app_telemetry.record_risk_rejection()
        if result.status == TradingCycleStatus.SIZING_REJECTED:
            self.container.app_telemetry.record_sizing_rejection()
        if result.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED:
            self.container.app_telemetry.record_authorization_rejection()

        # Check risk limits (Max failure bounds)
        if self._consecutive_failures >= self.policy.max_consecutive_cycle_failures:
            self.logger.error(
                f"Consecutive failures ({self._consecutive_failures}) hit threshold. Stopping session."
            )
            self.stop_session("Maximum consecutive cycle failures exceeded")

    def _handle_cycle_failure(self) -> None:
        with self._lock:
            self._failed_cycles += 1
            self._consecutive_failures += 1
        self.container.app_telemetry.record_cycle_failed()
        
        if self._consecutive_failures >= self.policy.max_consecutive_cycle_failures:
            self.stop_session("Maximum consecutive cycle failures exceeded")

    def _run_pre_flight_health_check(self) -> None:
        """Verifies policy settings and checks the initial health of all dependencies."""
        checks = self.container.health_monitor.check_all()
        
        # Verify required check dependencies according to options
        if self.policy.require_market_data_health:
            prov = checks.get("market_data_provider")
            if prov and prov.status == "UNHEALTHY":
                raise HealthCheckError(f"Market data provider unhealthy pre-flight: {prov.message}")
                
        if self.policy.require_model_health:
            inf = checks.get("inference")
            if inf and inf.status == "UNHEALTHY":
                raise HealthCheckError(f"Intelligence/Inference provider unhealthy pre-flight: {inf.message}")

        if self.policy.require_persistence:
            pers = checks.get("persistence")
            if pers and pers.status == "UNHEALTHY":
                raise HealthCheckError(f"Persistence framework unhealthy pre-flight: {pers.message}")

    def _verify_operational_health(self) -> None:
        try:
            self._run_pre_flight_health_check()
        except HealthCheckError as e:
            self.container.app_telemetry.record_component_health_failure()
            if self.policy.stop_on_persistence_failure or self.policy.require_persistence:
                self.logger.error(f"Health check failed during active cycle execution: {str(e)}")
                self.stop_session("Mid-session health check failure")

    def get_runtime_snapshot(self) -> IntegratedRuntimeSnapshot:
        """Retrieves and returns the full running status snapshot of active components."""
        p_state = self.container.portfolio_engine.get_state()
        
        health_checks = self.container.health_monitor.check_all()
        
        now = datetime.now(timezone.utc).isoformat()
        started_at_str = self.container.session.started_at

        # Construct position descriptions
        open_positions = []
        for symbol, pos in p_state.positions.items():
            if float(pos.quantity) != 0.0:
                open_positions.append({
                    "symbol": symbol,
                    "direction": pos.side.name,
                    "quantity": float(pos.quantity),
                    "entry_price": float(pos.average_entry_price),
                    "notional": float(pos.position_notional),
                    "unrealized_pnl": float(pos.unrealized_pnl)
                })

        return IntegratedRuntimeSnapshot(
            session_id=self.container.session.session_id,
            status=self.status,
            cycle_count=self._cycle_count,
            successful_cycles=self._successful_cycles,
            rejected_cycles=self._rejected_cycles,
            failed_cycles=self._failed_cycles,
            warmup_cycles=self._warmup_cycles,
            active_symbols=self.policy.enabled_symbols,
            started_at=started_at_str,
            updated_at=now,
            portfolio_equity=float(p_state.equity),
            available_balance=float(p_state.available_balance),
            open_positions=open_positions,
            last_cycle_id=None,
            last_cycle_status=None,
            component_health=health_checks,
            metadata=self.container.session.metadata
        )

    def _create_session_summary(self, reason: str) -> SessionSummary:
        p_state = self.container.portfolio_engine.get_state()
        
        now = datetime.now(timezone.utc).isoformat()
        started_at_str = self.container.session.started_at
        
        app_metrics = self.container.app_telemetry.get_metrics_snapshot()
        run_ms = app_metrics.get("avg_session_latency_ms", 0.0)

        # Positions opened and closed can be read from metrics or calculated
        positions_opened_count = app_metrics.get("positions_opened", 0)
        positions_closed_count = app_metrics.get("positions_closed", 0)

        return SessionSummary(
            session_id=self.container.session.session_id,
            started_at=started_at_str,
            completed_at=now,
            total_cycles=self._cycle_count,
            completed_cycles=self._successful_cycles,
            rejected_cycles=self._rejected_cycles,
            failed_cycles=self._failed_cycles,
            warmup_cycles=self._warmup_cycles,
            executions=app_metrics.get("orders_executed", 0),
            fills=app_metrics.get("fills_generated", 0),
            positions_opened=positions_opened_count,
            positions_closed=positions_closed_count,
            initial_equity=100000.0,  # Assumption
            final_equity=float(p_state.equity),
            realized_pnl=float(p_state.realized_pnl),
            unrealized_pnl=float(p_state.unrealized_pnl),
            total_fees=float(p_state.total_fees),
            max_drawdown=0.0,  # Can be computed if state has peak equity
            runtime_latency_ms=run_ms,
            stop_reason=reason,
            metadata={"telemetry_metrics": app_metrics}
        )
