import threading
import time
from typing import Dict, List, Optional, Any
from backend.persistence.policy import PersistencePolicy
from backend.persistence.telemetry import PersistenceTelemetry
from backend.persistence.exceptions import (
    UnsupportedPersistenceBackendError, AuditIntegrityError, PersistenceError
)
from backend.persistence.database.connection import db_session, begin_transaction
from backend.persistence.adapters.memory import (
    InMemoryRuntimeSessionRepository, InMemoryTradingCycleRepository, InMemoryDecisionRepository,
    InMemoryTradeProposalRepository, InMemoryRiskAuthorizationRepository, InMemoryPositionSizingRepository,
    InMemoryOrderIntentRepository, InMemoryExecutionRepository, InMemoryPortfolioRepository,
    InMemoryPositionLifecycleRepository, InMemoryAuditRepository
)
from backend.persistence.adapters.postgres import (
    PostgresRuntimeSessionRepository, PostgresTradingCycleRepository, PostgresDecisionRepository,
    PostgresTradeProposalRepository, PostgresRiskAuthorizationRepository, PostgresPositionSizingRepository,
    PostgresOrderIntentRepository, PostgresExecutionRepository, PostgresPortfolioRepository,
    PostgresPositionLifecycleRepository, PostgresAuditRepository
)
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord, DecisionRecord,
    TradeProposalRecord, RiskAuthorizationRecord, PositionSizingRecord,
    OrderIntentRecord, ExecutionRecord, FillRecord,
    PortfolioSnapshotRecord, PositionSnapshotRecord, PositionLifecycleRecord, AuditEventRecord
)


class PersistenceService:
    def __init__(self, policy: PersistencePolicy, telemetry: PersistenceTelemetry):
        self.policy = policy
        self.telemetry = telemetry
        self._memory_lock = threading.Lock()

        # In-memory repositories initialized once
        self._mem_session_repo = InMemoryRuntimeSessionRepository(self._memory_lock)
        self._mem_cycle_repo = InMemoryTradingCycleRepository(self._memory_lock)
        self._mem_decision_repo = InMemoryDecisionRepository(self._memory_lock)
        self._mem_proposal_repo = InMemoryTradeProposalRepository(self._memory_lock)
        self._mem_risk_repo = InMemoryRiskAuthorizationRepository(self._memory_lock)
        self._mem_sizing_repo = InMemoryPositionSizingRepository(self._memory_lock)
        self._mem_intent_repo = InMemoryOrderIntentRepository(self._memory_lock)
        self._mem_execution_repo = InMemoryExecutionRepository(self._memory_lock)
        self._mem_portfolio_repo = InMemoryPortfolioRepository(self._memory_lock)
        self._mem_lifecycle_repo = InMemoryPositionLifecycleRepository(self._memory_lock)
        self._mem_audit_repo = InMemoryAuditRepository(self._memory_lock)

    def _is_enabled(self) -> bool:
        return self.policy.persistence_enabled

    def _get_backend(self) -> str:
        return self.policy.persistence_backend

    # Dynamic repository factories helper
    def session_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresRuntimeSessionRepository(session)
        return self._mem_session_repo

    def cycle_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresTradingCycleRepository(session)
        return self._mem_cycle_repo

    def decision_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresDecisionRepository(session)
        return self._mem_decision_repo

    def proposal_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresTradeProposalRepository(session)
        return self._mem_proposal_repo

    def risk_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresRiskAuthorizationRepository(session)
        return self._mem_risk_repo

    def sizing_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresPositionSizingRepository(session)
        return self._mem_sizing_repo

    def intent_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresOrderIntentRepository(session)
        return self._mem_intent_repo

    def execution_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresExecutionRepository(session)
        return self._mem_execution_repo

    def portfolio_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresPortfolioRepository(session)
        return self._mem_portfolio_repo

    def lifecycle_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresPositionLifecycleRepository(session)
        return self._mem_lifecycle_repo

    def audit_repository(self, session=None):
        if self._get_backend() == "postgres":
            if session is None:
                raise UnsupportedPersistenceBackendError("Active DB session required for Postgres repository")
            return PostgresAuditRepository(session)
        return self._mem_audit_repo

    # Transaction scope helper
    def transaction(self):
        """Yields a database transaction context manager or a dummy context manager for in-memory."""
        if self._get_backend() == "postgres":
            return begin_transaction()
        # Dummy context manager for in-memory
        from contextlib import contextmanager
        @contextmanager
        def _dummy():
            yield None
        return _dummy()

    # Public transactional/audit persistence methods
    def save_session(self, record: RuntimeSessionRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.session_repository(session).save(record)
            self.telemetry.record_write("session", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("session", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def get_session(self, session_id: str) -> Optional[RuntimeSessionRecord]:
        if not self._is_enabled():
            return None
        t0 = time.perf_counter()
        try:
            if self._get_backend() == "postgres":
                with db_session() as session:
                    res = self.session_repository(session).get_by_id(session_id)
            else:
                res = self.session_repository().get_by_id(session_id)
            self.telemetry.record_read("session", "success", (time.perf_counter() - t0) * 1000.0)
            return res
        except Exception as e:
            self.telemetry.record_read("session", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_trading_cycle(self, record: TradingCycleRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.cycle_repository(session).save(record)
            self.telemetry.record_write("trading_cycle", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("trading_cycle", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def get_trading_cycle(self, cycle_id: str) -> Optional[TradingCycleRecord]:
        if not self._is_enabled():
            return None
        t0 = time.perf_counter()
        try:
            if self._get_backend() == "postgres":
                with db_session() as session:
                    res = self.cycle_repository(session).get_by_id(cycle_id)
            else:
                res = self.cycle_repository().get_by_id(cycle_id)
            self.telemetry.record_read("trading_cycle", "success", (time.perf_counter() - t0) * 1000.0)
            return res
        except Exception as e:
            self.telemetry.record_read("trading_cycle", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_decision(self, record: DecisionRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.decision_repository(session).save(record)
            self.telemetry.record_write("decision", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("decision", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_trade_proposal(self, record: TradeProposalRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.proposal_repository(session).save(record)
            self.telemetry.record_write("trade_proposal", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("trade_proposal", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_risk_authorization(self, record: RiskAuthorizationRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.risk_repository(session).save(record)
            self.telemetry.record_write("risk_authorization", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("risk_authorization", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_position_sizing(self, record: PositionSizingRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.sizing_repository(session).save(record)
            self.telemetry.record_write("position_sizing", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("position_sizing", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_order_intent(self, record: OrderIntentRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.intent_repository(session).save(record)
            self.telemetry.record_write("order_intent", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("order_intent", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_execution(self, record: ExecutionRecord, fills: List[FillRecord]) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.execution_repository(session).save(record, fills)
            self.telemetry.record_write("execution", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("execution", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_portfolio_snapshot(self, record: PortfolioSnapshotRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.portfolio_repository(session).save_snapshot(record)
            self.telemetry.record_write("portfolio_snapshot", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("portfolio_snapshot", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def save_position_lifecycle(self, record: PositionLifecycleRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            with self.transaction() as session:
                self.lifecycle_repository(session).save(record)
            self.telemetry.record_write("position_lifecycle", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("position_lifecycle", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def get_position_lifecycle(self, lifecycle_id: str) -> Optional[PositionLifecycleRecord]:
        if not self._is_enabled():
            return None
        t0 = time.perf_counter()
        try:
            if self._get_backend() == "postgres":
                with db_session() as session:
                    res = self.lifecycle_repository(session).get_by_id(lifecycle_id)
            else:
                res = self.lifecycle_repository().get_by_id(lifecycle_id)
            self.telemetry.record_read("position_lifecycle", "success", (time.perf_counter() - t0) * 1000.0)
            return res
        except Exception as e:
            self.telemetry.record_read("position_lifecycle", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    # Audit log validation/saving with chain checking
    def save_audit_event(self, record: AuditEventRecord) -> None:
        if not self._is_enabled():
            return
        t0 = time.perf_counter()
        try:
            # We serialize write transactions for audit logs to maintain strict order and prevent concurrency tampering/integrity gaps.
            with self.transaction() as session:
                audit_repo = self.audit_repository(session)
                last_event = audit_repo.get_latest_event(record.session_id)

                prev_hash = ""
                if last_event:
                    prev_hash = last_event.hash

                # Validates the event's supplied previous_hash
                if record.previous_hash != prev_hash:
                    raise AuditIntegrityError(
                        f"Audit Integrity Failure: Expected previous_hash '{prev_hash}', got '{record.previous_hash}'."
                    )

                # Dynamically calculate the signature to verify it is valid
                expected_hash = record.compute_hash()
                if record.hash and record.hash != expected_hash:
                    raise AuditIntegrityError(
                        f"Audit Integrity Failure: Hash mismatch. Record hash is '{record.hash}', expected '{expected_hash}'."
                    )

                # Construct new record using object.__setattr__ since it is frozen
                verified_record = record
                if not record.hash:
                    # Mutable override on frozen event to finalize canonical hash
                    object.__setattr__(record, "hash", expected_hash)

                audit_repo.save(verified_record)

            self.telemetry.record_write("audit_event", "success", (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            self.telemetry.record_write("audit_event", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e

    def get_audit_trail(self, session_id: str) -> List[AuditEventRecord]:
        if not self._is_enabled():
            return []
        t0 = time.perf_counter()
        try:
            if self._get_backend() == "postgres":
                with db_session() as session:
                    res = self.audit_repository(session).get_chain(session_id)
            else:
                res = self.audit_repository().get_chain(session_id)
            self.telemetry.record_read("audit_event", "success", (time.perf_counter() - t0) * 1000.0)
            return res
        except Exception as e:
            self.telemetry.record_read("audit_event", "failure", (time.perf_counter() - t0) * 1000.0)
            raise e
