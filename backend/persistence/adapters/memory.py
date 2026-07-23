import threading
import copy
from typing import Dict, List, Optional
from backend.persistence.exceptions import DuplicateRecordError
from backend.persistence.repositories import (
    RuntimeSessionRepository, TradingCycleRepository, DecisionRepository,
    TradeProposalRepository, RiskAuthorizationRepository, PositionSizingRepository,
    OrderIntentRepository, ExecutionRepository, PortfolioRepository,
    PositionLifecycleRepository, AuditRepository
)
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord, DecisionRecord,
    TradeProposalRecord, RiskAuthorizationRecord, PositionSizingRecord,
    OrderIntentRecord, ExecutionRecord, FillRecord,
    PortfolioSnapshotRecord, PositionLifecycleRecord, AuditEventRecord
)


class InMemoryRuntimeSessionRepository(RuntimeSessionRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, RuntimeSessionRecord] = {}

    def save(self, record: RuntimeSessionRecord) -> None:
        with self._lock:
            existing = self._store.get(record.session_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: Session {record.session_id} already exists."
                )
            self._store[record.session_id] = record

    def get_by_id(self, session_id: str) -> Optional[RuntimeSessionRecord]:
        with self._lock:
            return self._store.get(session_id)


class InMemoryTradingCycleRepository(TradingCycleRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, TradingCycleRecord] = {}

    def save(self, record: TradingCycleRecord) -> None:
        with self._lock:
            existing = self._store.get(record.cycle_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: TradingCycle {record.cycle_id} already exists."
                )
            self._store[record.cycle_id] = record

    def get_by_id(self, cycle_id: str) -> Optional[TradingCycleRecord]:
        with self._lock:
            return self._store.get(cycle_id)

    def list_by_session(self, session_id: str) -> List[TradingCycleRecord]:
        with self._lock:
            return [r for r in self._store.values() if r.session_id == session_id]


class InMemoryDecisionRepository(DecisionRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, DecisionRecord] = {}

    def save(self, record: DecisionRecord) -> None:
        with self._lock:
            existing = self._store.get(record.decision_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: Decision {record.decision_id} already exists."
                )
            self._store[record.decision_id] = record

    def get_by_id(self, decision_id: str) -> Optional[DecisionRecord]:
        with self._lock:
            return self._store.get(decision_id)

    def list_by_cycle(self, cycle_id: str) -> List[DecisionRecord]:
        with self._lock:
            return [r for r in self._store.values() if r.cycle_id == cycle_id]


class InMemoryTradeProposalRepository(TradeProposalRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, TradeProposalRecord] = {}

    def save(self, record: TradeProposalRecord) -> None:
        with self._lock:
            existing = self._store.get(record.proposal_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: TradeProposal {record.proposal_id} already exists."
                )
            self._store[record.proposal_id] = record

    def get_by_id(self, proposal_id: str) -> Optional[TradeProposalRecord]:
        with self._lock:
            return self._store.get(proposal_id)


class InMemoryRiskAuthorizationRepository(RiskAuthorizationRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, RiskAuthorizationRecord] = {}

    def save(self, record: RiskAuthorizationRecord) -> None:
        with self._lock:
            existing = self._store.get(record.authorization_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: RiskAuthorization {record.authorization_id} already exists."
                )
            self._store[record.authorization_id] = record

    def get_by_id(self, authorization_id: str) -> Optional[RiskAuthorizationRecord]:
        with self._lock:
            return self._store.get(authorization_id)


class InMemoryPositionSizingRepository(PositionSizingRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, PositionSizingRecord] = {}

    def save(self, record: PositionSizingRecord) -> None:
        with self._lock:
            existing = self._store.get(record.sizing_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: PositionSizing {record.sizing_id} already exists."
                )
            self._store[record.sizing_id] = record

    def get_by_id(self, sizing_id: str) -> Optional[PositionSizingRecord]:
        with self._lock:
            return self._store.get(sizing_id)


class InMemoryOrderIntentRepository(OrderIntentRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, OrderIntentRecord] = {}

    def save(self, record: OrderIntentRecord) -> None:
        with self._lock:
            existing = self._store.get(record.intent_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: OrderIntent {record.intent_id} already exists."
                )
            self._store[record.intent_id] = record

    def get_by_id(self, intent_id: str) -> Optional[OrderIntentRecord]:
        with self._lock:
            return self._store.get(intent_id)


class InMemoryExecutionRepository(ExecutionRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, ExecutionRecord] = {}
        self._fills_store: Dict[str, List[FillRecord]] = {}

    def save(self, record: ExecutionRecord, fills: List[FillRecord]) -> None:
        with self._lock:
            existing = self._store.get(record.execution_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: Execution {record.execution_id} already exists."
                )
            self._store[record.execution_id] = record
            self._fills_store[record.execution_id] = list(fills)

    def get_by_id(self, execution_id: str) -> Optional[ExecutionRecord]:
        with self._lock:
            return self._store.get(execution_id)

    def get_fills(self, execution_id: str) -> List[FillRecord]:
        with self._lock:
            return list(self._fills_store.get(execution_id, []))


class InMemoryPortfolioRepository(PortfolioRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, PortfolioSnapshotRecord] = {}

    def save_snapshot(self, snapshot: PortfolioSnapshotRecord) -> None:
        with self._lock:
            existing = self._store.get(snapshot.portfolio_snapshot_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: PortfolioSnapshot {snapshot.portfolio_snapshot_id} already exists."
                )
            self._store[snapshot.portfolio_snapshot_id] = snapshot

    def get_snapshot(self, portfolio_snapshot_id: str) -> Optional[PortfolioSnapshotRecord]:
        with self._lock:
            return self._store.get(portfolio_snapshot_id)

    def list_snapshots_by_portfolio(self, portfolio_id: str) -> List[PortfolioSnapshotRecord]:
        with self._lock:
            return [s for s in self._store.values() if s.portfolio_id == portfolio_id]


class InMemoryPositionLifecycleRepository(PositionLifecycleRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, PositionLifecycleRecord] = {}

    def save(self, record: PositionLifecycleRecord) -> None:
        with self._lock:
            existing = self._store.get(record.lifecycle_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: PositionLifecycle {record.lifecycle_id} already exists."
                )
            self._store[record.lifecycle_id] = record

    def get_by_id(self, lifecycle_id: str) -> Optional[PositionLifecycleRecord]:
        with self._lock:
            return self._store.get(lifecycle_id)

    def list_by_symbol(self, symbol: str) -> List[PositionLifecycleRecord]:
        with self._lock:
            return [r for r in self._store.values() if r.symbol == symbol]


class InMemoryAuditRepository(AuditRepository):
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._store: Dict[str, AuditEventRecord] = {}

    def save(self, record: AuditEventRecord) -> None:
        with self._lock:
            existing = self._store.get(record.audit_id)
            if existing is not None:
                raise DuplicateRecordError(
                    f"Conflict: AuditEvent {record.audit_id} already exists."
                )
            self._store[record.audit_id] = record

    def get_by_id(self, audit_id: str) -> Optional[AuditEventRecord]:
        with self._lock:
            return self._store.get(audit_id)

    def get_latest_event(self, session_id: str) -> Optional[AuditEventRecord]:
        chain = self.get_chain(session_id)
        return chain[-1] if chain else None

    def get_chain(self, session_id: str) -> List[AuditEventRecord]:
        with self._lock:
            events = [e for e in self._store.values() if e.session_id == session_id]
            # Order deterministically by timestamp, or stable sort
            return sorted(events, key=lambda e: e.timestamp)
