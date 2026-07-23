import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from backend.runtime.event_bus import BaseEventBus
from backend.runtime.events import TradingEvent
from backend.persistence.service import PersistenceService
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord, AuditEventRecord
)


class PersistenceEventHandler:
    """
    Subscribes to the runtime EventBus, transforming incoming event streams
    into persistent session, cycle, and cryptographic audit records.
    """
    def __init__(self, persistence_service: PersistenceService):
        self.persistence_service = persistence_service
        self._active_sessions: Dict[str, RuntimeSessionRecord] = {}
        self._active_cycles: Dict[str, Dict[str, Any]] = {}

    def register(self, event_bus: BaseEventBus) -> None:
        """Register handlers for all relevant trading runtime events."""
        event_types = [
            "RuntimeStarted", "RuntimeStopped", "RuntimePaused", "RuntimeResumed",
            "RuntimeFailed", "RuntimeStateChanged", "TradingCycleStarted",
            "TradingCycleFinished", "TradingCycleFailed", "DecisionCreated",
            "ProposalGenerated", "ProposalRejected", "RiskApproved", "RiskRejected",
            "PositionSized", "ExecutionAuthorized", "ExecutionRejected",
            "OrderExecuted", "PortfolioUpdated", "PositionOpened", "PositionClosed",
            "RuntimeError"
        ]
        for et in event_types:
            event_bus.subscribe(et, self.handle_event)

    def handle_event(self, event: TradingEvent) -> None:
        """Routes generic TradingEvent to specific persistence actions."""
        # 1. Update system lifecycle entities if they match
        self._process_entity_event(event)

        # 2. Build and persist a cryptographic audit ledger entry
        self._persist_audit_event(event)

    def _process_entity_event(self, event: TradingEvent) -> None:
        # Runtime session lifecycle tracking
        if event.event_type == "RuntimeStarted":
            record = RuntimeSessionRecord(
                session_id=event.session_id,
                status="STARTED",
                started_at=event.timestamp,
                metadata=event.metadata
            )
            self._active_sessions[event.session_id] = record
            self.persistence_service.save_session(record)

        elif event.event_type == "RuntimeStopped":
            existing = self._active_sessions.get(event.session_id)
            started_at = existing.started_at if existing else event.timestamp
            record = RuntimeSessionRecord(
                session_id=event.session_id,
                status="STOPPED",
                started_at=started_at,
                stopped_at=event.timestamp,
                metadata=event.metadata
            )
            if event.session_id in self._active_sessions:
                del self._active_sessions[event.session_id]
            self.persistence_service.save_session(record)

        elif event.event_type == "RuntimeFailed":
            existing = self._active_sessions.get(event.session_id)
            started_at = existing.started_at if existing else event.timestamp
            record = RuntimeSessionRecord(
                session_id=event.session_id,
                status="FAILED",
                started_at=started_at,
                stopped_at=event.timestamp,
                metadata=event.metadata
            )
            if event.session_id in self._active_sessions:
                del self._active_sessions[event.session_id]
            self.persistence_service.save_session(record)

        # Trading cycle progression tracking
        elif event.event_type == "TradingCycleStarted":
            assert event.cycle_id is not None
            # Extract cycle index if possible, e.g. "cycle-{session_id}-{index}"
            cycle_index = 0
            if "-" in event.cycle_id:
                try:
                    cycle_index = int(event.cycle_id.split("-")[-1])
                except ValueError:
                    pass

            self._active_cycles[event.cycle_id] = {
                "cycle_id": event.cycle_id,
                "session_id": event.session_id,
                "cycle_index": cycle_index,
                "status": "STARTED",
                "started_at": event.timestamp,
                "completed_at": event.timestamp,
                "latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "rejection_stage": None,
                "failed_stage": None,
                "rejection_reason": None,
                "fusion_id": None,
                "proposal_id": None,
                "risk_authorization_id": None,
                "sizing_id": None,
                "execution_authorization_id": None,
                "intent_id": None,
                "execution_id": None,
                "fill_ids": [],
                "intelligence_used": False,
                "proposal_generated": False,
                "risk_authorized": False,
                "execution_authorized": False,
                "executed": False,
                "portfolio_updated": False,
                "lifecycle_registered": False,
                "stage_timings": {},
                "policy_version": "",
                "metadata": event.metadata
            }

        elif event.cycle_id in self._active_cycles:
            cycle_id = event.cycle_id
            draft = self._active_cycles[cycle_id]

            if event.event_type == "DecisionCreated":
                draft["fusion_id"] = event.metadata.get("fusion_id")
                draft["intelligence_used"] = event.metadata.get("intel_used", False)

            elif event.event_type == "ProposalGenerated":
                draft["proposal_id"] = event.metadata.get("proposal_id")
                draft["proposal_generated"] = True

            elif event.event_type == "ProposalRejected":
                draft["rejection_stage"] = "FUSION"
                draft["rejection_reason"] = event.metadata.get("reason")

            elif event.event_type == "RiskApproved":
                draft["risk_authorization_id"] = event.metadata.get("risk_auth_id")
                draft["risk_authorized"] = True

            elif event.event_type == "RiskRejected":
                draft["rejection_stage"] = "RISK"
                draft["rejection_reason"] = event.metadata.get("reason")

            elif event.event_type == "PositionSized":
                draft["sizing_id"] = event.metadata.get("sizing_id")

            elif event.event_type == "ExecutionAuthorized":
                draft["execution_authorization_id"] = event.metadata.get("execution_auth_id")
                draft["intent_id"] = event.metadata.get("intent_id")
                draft["execution_authorized"] = True

            elif event.event_type == "ExecutionRejected":
                draft["rejection_stage"] = "EXECUTION_AUTHORIZATION"
                draft["rejection_reason"] = event.metadata.get("reason")

            elif event.event_type == "OrderExecuted":
                draft["execution_id"] = event.metadata.get("execution_id")
                draft["fill_ids"] = event.metadata.get("fill_ids", [])
                draft["executed"] = True

            elif event.event_type == "PortfolioUpdated":
                draft["portfolio_updated"] = True

            elif event.event_type in ("PositionOpened", "PositionClosed"):
                draft["lifecycle_registered"] = True

            elif event.event_type in ("TradingCycleFinished", "TradingCycleFailed"):
                # Complete details from final orchestrator notification
                draft["completed_at"] = event.timestamp
                draft["status"] = event.metadata.get("status", "COMPLETED")
                
                # Fetch timings and latency if attached
                draft["latency_ms"] = float(event.metadata.get("latency_ms", 0.0) or 0.0)
                draft["total_latency_ms"] = float(event.metadata.get("total_latency_ms", draft["latency_ms"]) or 0.0)
                
                # Error stage and reasons
                if event.event_type == "TradingCycleFailed":
                    draft["status"] = "FAILED"
                    draft["rejection_stage"] = event.metadata.get("stage", draft.get("rejection_stage"))
                    draft["failed_stage"] = event.metadata.get("stage", draft.get("rejection_stage"))
                    draft["rejection_reason"] = event.metadata.get("reason", event.metadata.get("error"))

                record = TradingCycleRecord(
                    cycle_id=draft["cycle_id"],
                    session_id=draft["session_id"],
                    cycle_index=draft["cycle_index"],
                    status=draft["status"],
                    started_at=draft["started_at"],
                    completed_at=draft["completed_at"],
                    latency_ms=draft["latency_ms"],
                    total_latency_ms=draft["total_latency_ms"],
                    rejection_stage=draft["rejection_stage"],
                    failed_stage=draft["failed_stage"],
                    rejection_reason=draft["rejection_reason"],
                    fusion_id=draft["fusion_id"],
                    proposal_id=draft["proposal_id"],
                    risk_authorization_id=draft["risk_authorization_id"],
                    sizing_id=draft["sizing_id"],
                    execution_authorization_id=draft["execution_authorization_id"],
                    intent_id=draft["intent_id"],
                    execution_id=draft["execution_id"],
                    fill_ids=draft["fill_ids"],
                    intelligence_used=draft["intelligence_used"],
                    proposal_generated=draft["proposal_generated"],
                    risk_authorized=draft["risk_authorized"],
                    execution_authorized=draft["execution_authorized"],
                    executed=draft["executed"],
                    portfolio_updated=draft["portfolio_updated"],
                    lifecycle_registered=draft["lifecycle_registered"],
                    stage_timings=draft.get("stage_timings", {}),
                    policy_version=draft.get("policy_version", ""),
                    metadata=draft.get("metadata", {})
                )
                self.persistence_service.save_trading_cycle(record)
                
                # Clean up memory trace
                del self._active_cycles[cycle_id]

    def _persist_audit_event(self, event: TradingEvent) -> None:
        """Constructs and cryptographically signs a persistent audit logging event."""
        # Class maps to logical entity types
        entity_mapping = {
            "RuntimeStarted": ("session", event.session_id),
            "RuntimeStopped": ("session", event.session_id),
            "RuntimePaused": ("session", event.session_id),
            "RuntimeResumed": ("session", event.session_id),
            "RuntimeFailed": ("session", event.session_id),
            "RuntimeStateChanged": ("session", event.session_id),
            "TradingCycleStarted": ("cycle", event.cycle_id or ""),
            "TradingCycleFinished": ("cycle", event.cycle_id or ""),
            "TradingCycleFailed": ("cycle", event.cycle_id or ""),
            "DecisionCreated": ("decision", event.metadata.get("fusion_id", "")),
            "ProposalGenerated": ("proposal", event.metadata.get("proposal_id", "")),
            "ProposalRejected": ("proposal", event.metadata.get("reason", "")),
            "RiskApproved": ("risk", event.metadata.get("risk_auth_id", "")),
            "RiskRejected": ("risk", event.metadata.get("reason", "")),
            "PositionSized": ("sizing", event.metadata.get("sizing_id", "")),
            "ExecutionAuthorized": ("execution_authorization", event.metadata.get("execution_auth_id", "")),
            "ExecutionRejected": ("execution_authorization", event.metadata.get("reason", "")),
            "OrderExecuted": ("execution", event.metadata.get("execution_id", "")),
            "PortfolioUpdated": ("portfolio", event.session_id),
            "PositionOpened": ("position", event.metadata.get("symbol", "")),
            "PositionClosed": ("position", event.metadata.get("symbol", "")),
            "RuntimeError": ("error", event.event_id)
        }

        entity_type, entity_id = entity_mapping.get(
            event.event_type, ("generic", event.event_id)
        )

        symbol = event.metadata.get("symbol")
        source = event.metadata.get("source", "trading_runtime")
        status = event.metadata.get("status", "SUCCESS")
        if event.event_type in ("RuntimeFailed", "TradingCycleFailed", "RuntimeError"):
            status = "FAILED"

        # Lookup previous hash from previous audit logged records
        prev_hash = ""
        if self.persistence_service._is_enabled():
            last = self.persistence_service.audit_repository().get_latest_event(event.session_id)
            if last:
                prev_hash = last.hash

        # Extract nested payload
        payload = {
            "metadata": event.metadata,
            "runtime_id": event.runtime_id
        }
        if hasattr(event, "old_state") and hasattr(event, "new_state"):
            payload["old_state"] = getattr(event, "old_state")
            payload["new_state"] = getattr(event, "new_state")
        if hasattr(event, "error_msg"):
            payload["error_msg"] = getattr(event, "error_msg")

        audit_record = AuditEventRecord(
            audit_id=str(uuid.uuid4()),
            session_id=event.session_id,
            cycle_id=event.cycle_id,
            event_type=event.event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            symbol=symbol,
            timestamp=event.timestamp,
            source_component=source,
            status=status,
            payload=payload,
            previous_hash=prev_hash
        )

        self.persistence_service.save_audit_event(audit_record)
