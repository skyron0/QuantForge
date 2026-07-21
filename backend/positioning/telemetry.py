import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from backend.positioning.models import PositionSizeResult, PositionSizingContext

logger = logging.getLogger("QuantForge.PositionSizingTelemetry")


class PositionSizingTelemetrySink(ABC):
    """Abstract base class for telemetry tracking of position sizing decisions."""

    @abstractmethod
    def record(
        self,
        result: Optional[PositionSizeResult],
        context: PositionSizingContext,
        success: bool,
        rejection_reason: str,
        latency_ms: float,
    ) -> None:
        """Record a position sizing attempt."""
        pass


class ConsolePositionSizingTelemetrySink(PositionSizingTelemetrySink):
    """Console-logging implementation of the position sizing telemetry sink."""

    def record(
        self,
        result: Optional[PositionSizeResult],
        context: PositionSizingContext,
        success: bool,
        rejection_reason: str,
        latency_ms: float,
    ) -> None:
        if success and result:
            logger.info(
                f"[PositionSizing APPROVED] ID: {result.sizing_id} | Proposal: {result.proposal_id} | "
                f"Symbol: {result.symbol} | Direction: {result.direction} | "
                f"Qty: {result.quantity:.6f} | Notional: {result.position_notional:.2f} | "
                f"Risk Amt: {result.risk_amount:.2f} (Auth: {result.authorized_risk_fraction * 100:.2f}%) | "
                f"Est. Margin: {result.estimated_margin_required:.2f} @ {result.leverage}x | "
                f"Latency: {latency_ms:.2f}ms"
            )
        else:
            logger.warning(
                f"[PositionSizing REJECTED] Symbol: {context.symbol} | "
                f"Reason: {rejection_reason} | Latency: {latency_ms:.2f}ms"
            )
