import time
from abc import ABC, abstractmethod
from backend.portfolio.models import PortfolioState

class PortfolioTelemetrySink(ABC):
    @abstractmethod
    def record_update(
        self,
        state: PortfolioState,
        latency_ms: float,
        status: str,
        rejection_reason: str = ""
    ):
        """Records a portfolio state update event."""
        pass

class ConsolePortfolioTelemetrySink(PortfolioTelemetrySink):
    def record_update(
        self,
        state: PortfolioState,
        latency_ms: float,
        status: str,
        rejection_reason: str = ""
    ):
        print(
            f"[PORTFOLIO TELEMETRY] [{state.timestamp}] "
            f"State updated: status={status}, equity={state.equity:.4f}, "
            f"cash={state.cash_balance:.4f}, margin={state.used_margin:.4f}, "
            f"exposure(gross/net)={state.gross_exposure:.4f}/{state.net_exposure:.4f}, "
            f"positions={state.open_position_count}, latency={latency_ms:.3f}ms"
        )
        if rejection_reason:
            print(f"  └─ Rejection / Warning: {rejection_reason}")
