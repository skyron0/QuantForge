import abc
import logging
from typing import Any, Dict, Optional
from backend.portfolio.models import PositionSide
from backend.position_lifecycle.models import PositionLifecycleStatus, ProtectiveTriggerType

logger = logging.getLogger("QuantForge.PositionLifecycle")


class PositionLifecycleTelemetrySink(abc.ABC):
    @abc.abstractmethod
    def record_evaluation(
        self,
        lifecycle_id: str,
        position_id: str,
        symbol: str,
        position_side: PositionSide,
        market_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        active_trailing_stop: Optional[float],
        highest_price_since_entry: Optional[float],
        lowest_price_since_entry: Optional[float],
        trigger_type: Optional[ProtectiveTriggerType],
        exit_proposal_generated: bool,
        lifecycle_status: PositionLifecycleStatus,
        policy_version: str,
        latency_ms: float,
        rejection_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records protective evaluation telemetry metrics."""
        pass


class ConsolePositionLifecycleTelemetrySink(PositionLifecycleTelemetrySink):
    def record_evaluation(
        self,
        lifecycle_id: str,
        position_id: str,
        symbol: str,
        position_side: PositionSide,
        market_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        active_trailing_stop: Optional[float],
        highest_price_since_entry: Optional[float],
        lowest_price_since_entry: Optional[float],
        trigger_type: Optional[ProtectiveTriggerType],
        exit_proposal_generated: bool,
        lifecycle_status: PositionLifecycleStatus,
        policy_version: str,
        latency_ms: float,
        rejection_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        trig = trigger_type.value if trigger_type else "NONE"
        status_val = lifecycle_status.value
        
        log_msg = (
            f"[PositionLifecycle] ID: {lifecycle_id} | Pos: {position_id} | Sym: {symbol} | "
            f"Side: {position_side.value} | Price: {market_price:.4f} | SL: {stop_loss} | "
            f"TP: {take_profit} | Trailing: {active_trailing_stop} | Trigger: {trig} | "
            f"ExitGen: {exit_proposal_generated} | Status: {status_val} | "
            f"Latency: {latency_ms:.3f}ms"
        )
        if rejection_reason:
            log_msg += f" | Rejection: {rejection_reason}"
            
        logger.info(log_msg)
