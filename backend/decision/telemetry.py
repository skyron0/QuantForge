import logging
from abc import ABC, abstractmethod
from typing import Optional
from backend.decision.models import FusionResult, TradeProposal

logger = logging.getLogger("quantforge.decision.telemetry")


class DecisionTelemetrySink(ABC):
    """
    Abstract interface for recording safe, audited decision fusion metadata.
    """

    @abstractmethod
    def record(
        self,
        result: FusionResult,
        proposal: Optional[TradeProposal],
        latency_ms: float,
        rejection_reason: Optional[str] = None,
    ) -> None:
        pass


class ConsoleDecisionTelemetrySink(DecisionTelemetrySink):
    """
    A telemetry sink that outputs structured, audit-safe logs via standard Python logging.
    """

    def record(
        self,
        result: FusionResult,
        proposal: Optional[TradeProposal],
        latency_ms: float,
        rejection_reason: Optional[str] = None,
    ) -> None:
        log_payload = {
            "fusion_id": result.fusion_id,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "policy_version": result.policy_version,
            "ml_model_version": result.source_model_version,
            "intelligence_used": result.intelligence_used,
            "intelligence_age_seconds": result.intelligence_age_seconds,
            "agreement_score": result.agreement_score,
            "fusion_score": result.fusion_score,
            "resulting_direction": result.direction,
            "proposal_generated": proposal is not None,
            "rejection_reason": rejection_reason or "None",
            "latency_ms": latency_ms,
        }

        # Safe logging without any credentials
        logger.info(f"[Decision Telemetry Audit] {log_payload}")
