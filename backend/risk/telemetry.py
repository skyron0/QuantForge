from abc import ABC, abstractmethod
import logging
from backend.risk.models import RiskAuthorizationResult, RiskContext


class RiskTelemetrySink(ABC):

    @abstractmethod
    def record(
        self,
        result: RiskAuthorizationResult,
        context: RiskContext,
        latency_ms: float,
    ) -> None:
        pass


class ConsoleRiskTelemetrySink(RiskTelemetrySink):

    def __init__(self, logger_name: str = "QuantForge.RiskTelemetry"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def record(
        self,
        result: RiskAuthorizationResult,
        context: RiskContext,
        latency_ms: float,
    ) -> None:
        self.logger.info(
            f"[RiskGuard] evaluated={result.evaluated_at} | symbol={result.symbol} | "
            f"status={result.status.value} | "
            f"authorized_risk={result.authorized_risk_fraction:.6f} | latency={latency_ms:.3f}ms | "
            f"rules={result.triggered_rules} | rejections={result.rejection_reasons}"
        )
