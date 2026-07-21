import logging
from abc import ABC, abstractmethod
from backend.execution_adapter.models import ExecutionResult

logger = logging.getLogger("QuantForge.ExecutionAdapter")

class ExecutionTelemetrySink(ABC):
    @abstractmethod
    def record(self, result: ExecutionResult, latency_ms: float) -> None:
        pass

class ConsoleExecutionTelemetrySink(ExecutionTelemetrySink):
    def record(self, result: ExecutionResult, latency_ms: float) -> None:
        # Structured log print
        msg = (
            f"Execution {result.status.value}: ID {result.execution_id} | "
            f"Intent {result.intent_id} | Proposal {result.proposal_id} | "
            f"Symbol {result.symbol} | Direction {result.direction.value} | "
            f"Qty {result.filled_quantity}/{result.requested_quantity} | "
            f"Avg Price {result.average_fill_price:.4f} | Fees {result.total_fees:.4f} | "
            f"Slippage {result.total_slippage:.4f} | Latency {latency_ms:.2f}ms"
        )
        if result.rejection_reason:
            msg += f" | Rejection Reason: {result.rejection_reason}"
        
        if result.status in ("REJECTED", "EXPIRED"):
            logger.warning(msg)
        else:
            logger.info(msg)
