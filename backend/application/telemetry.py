import logging
import threading
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class IntegratedRuntimeTelemetrySink:
    """
    Thread-safe telemetry accumulator tracking paper trading session aggregation-level metrics.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        
        # Session metrics
        self.sessions_started = 0
        self.sessions_completed = 0
        self.sessions_failed = 0
        
        # Cycle metrics
        self.cycles_started = 0
        self.cycles_completed = 0
        self.cycles_rejected = 0
        self.cycles_failed = 0
        self.cycles_warmup = 0
        
        # Pipeline stage decisions
        self.signals_generated = 0
        self.proposals_generated = 0
        self.risk_rejections = 0
        self.sizing_rejections = 0
        self.authorization_rejections = 0
        
        # Execution metrics
        self.orders_executed = 0
        self.fills_generated = 0
        
        # Position lifecycle metrics
        self.positions_opened = 0
        self.positions_closed = 0
        self.protective_exits = 0
        
        # Infrastructure failures
        self.persistence_failures = 0
        self.component_health_failures = 0
        
        # Latency lists
        self.cycle_latencies_ms: List[float] = []
        self.session_latencies_ms: List[float] = []

    def record_session_start(self) -> None:
        with self._lock:
            self.sessions_started += 1

    def record_session_complete(self, latency_ms: float) -> None:
        with self._lock:
            self.sessions_completed += 1
            self.session_latencies_ms.append(latency_ms)

    def record_session_failed(self, latency_ms: float) -> None:
        with self._lock:
            self.sessions_failed += 1
            self.session_latencies_ms.append(latency_ms)

    def record_cycle_start(self) -> None:
        with self._lock:
            self.cycles_started += 1

    def record_cycle_complete(self, latency_ms: float) -> None:
        with self._lock:
            self.cycles_completed += 1
            self.cycle_latencies_ms.append(latency_ms)

    def record_cycle_reject(self) -> None:
        with self._lock:
            self.cycles_rejected += 1

    def record_cycle_failed(self) -> None:
        with self._lock:
            self.cycles_failed += 1

    def record_cycle_warmup(self) -> None:
        with self._lock:
            self.cycles_warmup += 1

    def record_signal_generated(self) -> None:
        with self._lock:
            self.signals_generated += 1

    def record_proposal_generated(self) -> None:
        with self._lock:
            self.proposals_generated += 1

    def record_risk_rejection(self) -> None:
        with self._lock:
            self.risk_rejections += 1

    def record_sizing_rejection(self) -> None:
        with self._lock:
            self.sizing_rejections += 1

    def record_authorization_rejection(self) -> None:
        with self._lock:
            self.authorization_rejections += 1

    def record_order_executed(self) -> None:
        with self._lock:
            self.orders_executed += 1

    def record_fill_generated(self) -> None:
        with self._lock:
            self.fills_generated += 1

    def record_position_opened(self) -> None:
        with self._lock:
            self.positions_opened += 1

    def record_position_closed(self) -> None:
        with self._lock:
            self.positions_closed += 1

    def record_protective_exit(self) -> None:
        with self._lock:
            self.protective_exits += 1

    def record_persistence_failure(self) -> None:
        with self._lock:
            self.persistence_failures += 1

    def record_component_health_failure(self) -> None:
        with self._lock:
            self.component_health_failures += 1

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg_cycle_latency = (
                sum(self.cycle_latencies_ms) / len(self.cycle_latencies_ms)
                if self.cycle_latencies_ms else 0.0
            )
            avg_session_latency = (
                sum(self.session_latencies_ms) / len(self.session_latencies_ms)
                if self.session_latencies_ms else 0.0
            )
            return {
                "sessions_started": self.sessions_started,
                "sessions_completed": self.sessions_completed,
                "sessions_failed": self.sessions_failed,
                "cycles_started": self.cycles_started,
                "cycles_completed": self.cycles_completed,
                "cycles_rejected": self.cycles_rejected,
                "cycles_failed": self.cycles_failed,
                "cycles_warmup": self.cycles_warmup,
                "signals_generated": self.signals_generated,
                "proposals_generated": self.proposals_generated,
                "risk_rejections": self.risk_rejections,
                "sizing_rejections": self.sizing_rejections,
                "authorization_rejections": self.authorization_rejections,
                "orders_executed": self.orders_executed,
                "fills_generated": self.fills_generated,
                "positions_opened": self.positions_opened,
                "positions_closed": self.positions_closed,
                "protective_exits": self.protective_exits,
                "persistence_failures": self.persistence_failures,
                "component_health_failures": self.component_health_failures,
                "avg_cycle_latency_ms": avg_cycle_latency,
                "avg_session_latency_ms": avg_session_latency,
            }


class ConsoleIntegratedRuntimeTelemetrySink(IntegratedRuntimeTelemetrySink):
    """
    Extension of telemetry sink that logs state changes to standard logger printouts.
    """
    def __init__(self, logger_name: str = "QuantForge.RuntimeApplication") -> None:
        super().__init__()
        self.logger = logging.getLogger(logger_name)

    def record_session_start(self) -> None:
        super().record_session_start()
        self.logger.info("Session started.")

    def record_session_complete(self, latency_ms: float) -> None:
        super().record_session_complete(latency_ms)
        self.logger.info(f"Session completed successfully in {latency_ms:.2f}ms.")

    def record_session_failed(self, latency_ms: float) -> None:
        super().record_session_failed(latency_ms)
        self.logger.error(f"Session failed in {latency_ms:.2f}ms.")

    def record_cycle_start(self) -> None:
        super().record_cycle_start()

    def record_cycle_complete(self, latency_ms: float) -> None:
        super().record_cycle_complete(latency_ms)

    def record_cycle_reject(self) -> None:
        super().record_cycle_reject()
        self.logger.warning("Cycle rejected at downstream checks.")

    def record_cycle_failed(self) -> None:
        super().record_cycle_failed()
        self.logger.error("Cycle failed to complete.")

    def record_cycle_warmup(self) -> None:
        super().record_cycle_warmup()
        self.logger.info("Cycle warm-up execution.")

    def record_signal_generated(self) -> None:
        super().record_signal_generated()

    def record_proposal_generated(self) -> None:
        super().record_proposal_generated()

    def record_risk_rejection(self) -> None:
        super().record_risk_rejection()
        self.logger.warning("Risk guard evaluation rejected.")

    def record_sizing_rejection(self) -> None:
        super().record_sizing_rejection()
        self.logger.warning("Position sizing rejected.")

    def record_authorization_rejection(self) -> None:
        super().record_authorization_rejection()
        self.logger.warning("Execution authorization engine rejected.")

    def record_order_executed(self) -> None:
        super().record_order_executed()

    def record_fill_generated(self) -> None:
        super().record_fill_generated()

    def record_position_opened(self) -> None:
        super().record_position_opened()
        self.logger.info("Position opened successfully.")

    def record_position_closed(self) -> None:
        super().record_position_closed()
        self.logger.info("Position closed successfully.")

    def record_protective_exit(self) -> None:
        super().record_protective_exit()
        self.logger.info("Protective exit triggered by PositionLifecycle.")

    def record_persistence_failure(self) -> None:
        super().record_persistence_failure()
        self.logger.error("Infrastructure Persistence Failure detected.")

    def record_component_health_failure(self) -> None:
        super().record_component_health_failure()
        self.logger.error("Component health failure detected.")
