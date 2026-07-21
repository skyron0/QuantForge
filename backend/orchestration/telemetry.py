import logging
from abc import ABC, abstractmethod
from backend.orchestration.models import TradingCycleResult


class TradingCycleTelemetrySink(ABC):
    """Abstract interface for recording trading cycle telemetry."""
    @abstractmethod
    def record_cycle(self, result: TradingCycleResult) -> None:
        """Record the telemetry of a completed or failed cycle."""
        pass


class ConsoleTradingCycleTelemetrySink(TradingCycleTelemetrySink):
    """Logs trading cycle telemetry to standard Python logging."""
    def __init__(self, logger_name: str = "TradingCycleOrchestrator") -> None:
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def record_cycle(self, result: TradingCycleResult) -> None:
        msg = (
            f"Cycle ID: {result.cycle_id} | Symbol: {result.symbol} | "
            f"Status: {result.status.value} | Reached Stages: "
            f"[intel={result.intelligence_used}, prop={result.proposal_generated}, "
            f"risk={result.risk_authorized}, exec_auth={result.execution_authorized}, "
            f"exec={result.executed}, portfolio={result.portfolio_updated}, "
            f"lifecycle={result.lifecycle_registered}] | "
            f"Latency: {result.latency_ms:.2f}ms"
        )
        if result.rejection_stage:
            msg += f" | Rejected at: {result.rejection_stage} (Reason: {result.rejection_reason})"
            
        self.logger.info(msg)
