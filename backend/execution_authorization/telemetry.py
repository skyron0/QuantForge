import logging
from abc import ABC, abstractmethod
from typing import Dict, Any


class ExecutionAuthorizationTelemetrySink(ABC):
    """
    Abstract interface for recording and auditing execution security and authorization metrics.
    Handles events such as authorization approval, rejection, and system checks.
    """

    @abstractmethod
    def record_authorization(self, event: Dict[str, Any]) -> None:
        """
        Persists security/audit log entries for an authorization request.
        """
        pass


class ConsoleExecutionAuthorizationTelemetrySink(ExecutionAuthorizationTelemetrySink):
    """
    Standard console logging implementation for ExecutionAuthorization telemetry.
    """

    def __init__(self, logger_name: str = "QuantForge.ExecutionAuthorization"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def record_authorization(self, event: Dict[str, Any]) -> None:
        status = event.get("status")
        symbol = event.get("symbol")
        direction = event.get("direction")
        qty = event.get("quantity")
        latency = event.get("latency_ms")
        proposal_id = event.get("proposal_id")

        if status == "AUTHORIZED":
            intent_id = event.get("intent_id")
            self.logger.info(
                f"Execution AUTHORIZED: Intent {intent_id} | Proposal {proposal_id} | "
                f"Symbol {symbol} | Direction {direction} | Qty {qty} | Latency {latency:.2f}ms"
            )
        else:
            reason = event.get("rejection_reason")
            triggered = event.get("triggered_rules")
            self.logger.warning(
                f"Execution REJECTED: Proposal {proposal_id} | Symbol {symbol} | "
                f"Reason: '{reason}' | Rules Triggered: {triggered} | Latency {latency:.2f}ms"
            )
