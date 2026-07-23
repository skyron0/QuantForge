import os
import ast
import pytest
import threading
from decimal import Decimal
from datetime import datetime, timezone
from backend.persistence.exceptions import (
    PersistenceValidationError, DuplicateRecordError, AuditIntegrityError, UnsupportedPersistenceBackendError
)
from backend.persistence.policy import PersistencePolicy
from backend.persistence.telemetry import PersistenceTelemetry
from backend.persistence.service import PersistenceService
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord,
    TradeProposalRecord, RiskAuthorizationRecord, PositionSizingRecord,
    OrderIntentRecord, ExecutionRecord, FillRecord,
    PortfolioSnapshotRecord, PositionSnapshotRecord, PositionLifecycleRecord, AuditEventRecord
)
from backend.persistence.adapters.memory import (
    InMemoryRuntimeSessionRepository, InMemoryTradingCycleRepository, InMemoryAuditRepository
)


@pytest.fixture
def base_policy() -> PersistencePolicy:
    return PersistencePolicy(
        persistence_enabled=True,
        persistence_backend="memory",
        database_pool_size=5,
        database_pool_overflow=10,
        database_timeout_seconds=5.0
    )


@pytest.fixture
def dev_telemetry() -> PersistenceTelemetry:
    return PersistenceTelemetry(enabled=True)


@pytest.fixture
def persistence_service(base_policy, dev_telemetry) -> PersistenceService:
    return PersistenceService(policy=base_policy, telemetry=dev_telemetry)


# 1. Model Timestamp and Field Validation
def test_session_record_utc_timestamp_validation():
    # Valid ISO UTC timestamp
    rec = RuntimeSessionRecord(
        session_id="session-1",
        status="STARTED",
        started_at="2026-07-22T12:00:00Z"
    )
    assert rec.started_at == "2026-07-22T12:00:00+00:00"


    # Naive timestamp should reject
    with pytest.raises(PersistenceValidationError):
        RuntimeSessionRecord(
            session_id="session-1",
            status="STARTED",
            started_at="2026-07-22 12:00:00"
        )

    # Empty session_id should reject
    with pytest.raises(PersistenceValidationError):
        RuntimeSessionRecord(
            session_id="",
            status="STARTED",
            started_at="2026-07-22T12:00:00Z"
        )


# 2. Decimal Precision Safety Checks (Post-Initialization Conversions)
def test_sizing_record_decimal_conversion():
    # Pass float/string fields to check that dataclass converts them safely to Decimal
    rec = PositionSizingRecord(
        sizing_id="sizing-1",
        proposal_id="prop-1",
        authorization_id="auth-1",
        symbol="ETH/USDT",
        direction="BUY",
        quantity=0.5,  # type: ignore
        position_notional="1500.00",  # type: ignore
        entry_price=Decimal("3000.00"),  # Decimal
        stop_loss_price=2900,  # type: ignore
        stop_distance_absolute=100.0,  # type: ignore
        stop_distance_fraction=0.033,  # type: ignore
        authorized_risk_fraction=0.01,  # type: ignore
        risk_amount=15.0,  # type: ignore
        leverage=1.0,  # type: ignore
        estimated_margin_required=1500.0,  # type: ignore
        policy_version="v1.0",
        created_at="2026-07-22T12:00:00Z",
        source_model_version="reasoning-1"
    )
    assert isinstance(rec.quantity, Decimal)
    assert rec.quantity == Decimal("0.5")
    assert isinstance(rec.position_notional, Decimal)
    assert rec.position_notional == Decimal("1500.00")
    assert isinstance(rec.entry_price, Decimal)
    assert rec.stop_loss_price == Decimal("2900")


# 3. Duplicate Record Prevention
def test_in_memory_repository_duplicate_prevention():
    lock = threading.Lock()
    repo = InMemoryRuntimeSessionRepository(lock)
    rec1 = RuntimeSessionRecord(
        session_id="sess-1",
        status="STARTED",
        started_at="2026-07-22T12:00:00Z"
    )
    repo.save(rec1)
    
    # Saving duplicate record should throw DuplicateRecordError
    with pytest.raises(DuplicateRecordError):
        repo.save(rec1)


# 4. In-Memory Save and Retrieve
def test_in_memory_retrieve_by_id():
    lock = threading.Lock()
    repo = InMemoryTradingCycleRepository(lock)
    rec = TradingCycleRecord(
        cycle_id="cycle-1",
        session_id="sess-1",
        cycle_index=1,
        status="COMPLETED",
        started_at="2026-07-22T12:00:00Z",
        completed_at="2026-07-22T12:00:01Z",
        latency_ms=10.0,
        total_latency_ms=10.0
    )
    repo.save(rec)
    fetched = repo.get_by_id("cycle-1")
    assert fetched is not None
    assert fetched.cycle_id == "cycle-1"
    assert fetched.session_id == "sess-1"

    assert repo.get_by_id("non-existent") is None


# 5. Telemetry Metric Ingestion
def test_persistence_telemetry_tracking(persistence_service, dev_telemetry):
    rec = RuntimeSessionRecord(
        session_id="sess-telemetry",
        status="STARTED",
        started_at="2026-07-22T12:00:00Z"
    )
    
    persistence_service.save_session(rec)
    metrics = dev_telemetry.get_metrics()
    
    assert metrics["write_count"] == 1
    assert "session" in metrics["table_write_latencies"]
    assert len(metrics["table_write_latencies"]["session"]) == 1
    
    persistence_service.get_session("sess-telemetry")
    metrics = dev_telemetry.get_metrics()
    assert metrics["read_count"] == 1


# 6. SHA-256 Audit Trail Cryptographic Hash Calculation
def test_audit_event_hash_calculation():
    payload = {"test": 123, "amount": Decimal("10.5")}
    evt = AuditEventRecord(
        audit_id="audit-1",
        session_id="session-hash",
        cycle_id="cycle-1",
        event_type="TradingCycleStarted",
        entity_type="cycle",
        entity_id="cycle-1",
        symbol="BTC/USDT",
        timestamp="2026-07-22T12:00:00Z",
        source_component="trading_runtime",
        status="SUCCESS",
        payload=payload,
        previous_hash="prev-hash-123"
    )
    h = evt.compute_hash()
    assert len(h) == 64  # SHA-256 is 64 characters long hex
    
    # Hash calculation must be deterministic: same values yield same hash
    evt2 = AuditEventRecord(
        audit_id="audit-1",
        session_id="session-hash",
        cycle_id="cycle-1",
        event_type="TradingCycleStarted",
        entity_type="cycle",
        entity_id="cycle-1",
        symbol="BTC/USDT",
        timestamp="2026-07-22T12:00:00Z",
        source_component="trading_runtime",
        status="SUCCESS",
        payload=payload,
        previous_hash="prev-hash-123"
    )
    assert evt2.compute_hash() == h


# 7. Audit Chain Integrity Preservation
def test_audit_chain_verification_and_integrity_gate(persistence_service):
    # Enable hashing/audit
    evt1 = AuditEventRecord(
        audit_id="audit-event-1",
        session_id="sess-audit-chain",
        cycle_id="cycle-1",
        event_type="RuntimeStarted",
        entity_type="session",
        entity_id="sess-audit-chain",
        symbol=None,
        timestamp="2026-07-22T12:00:00Z",
        source_component="runtime",
        status="SUCCESS",
        payload={}
    )
    persistence_service.save_audit_event(evt1)
    
    # The signature gets dynamically assigned & saved
    fetched_chain = persistence_service.get_audit_trail("sess-audit-chain")
    assert len(fetched_chain) == 1
    h1 = fetched_chain[0].hash
    assert h1 != ""

    # Second event should chain with expected previous_hash equal to h1
    evt2 = AuditEventRecord(
        audit_id="audit-event-2",
        session_id="sess-audit-chain",
        cycle_id="cycle-1",
        event_type="TradingCycleStarted",
        entity_type="cycle",
        entity_id="cycle-1",
        symbol="BTC/USDT",
        timestamp="2026-07-22T12:00:01Z",
        source_component="runtime",
        status="SUCCESS",
        payload={},
        previous_hash=h1
    )
    persistence_service.save_audit_event(evt2)
    
    fetched_chain = persistence_service.get_audit_trail("sess-audit-chain")
    assert len(fetched_chain) == 2
    assert fetched_chain[1].previous_hash == h1

    # Tampered event with invalid previous_hash should be rejected
    evt3_tampered = AuditEventRecord(
        audit_id="audit-event-3",
        session_id="sess-audit-chain",
        cycle_id="cycle-1",
        event_type="TradingCycleFinished",
        entity_type="cycle",
        entity_id="cycle-1",
        symbol="BTC/USDT",
        timestamp="2026-07-22T12:00:02Z",
        source_component="runtime",
        status="SUCCESS",
        payload={},
        previous_hash="tampered-hash-does-not-match"
    )
    with pytest.raises(AuditIntegrityError):
        persistence_service.save_audit_event(evt3_tampered)


# 8. AST Isolation check: zero coupling between engines/adaptors and database scheme
def test_persistence_package_database_isolation():
    forbidden_keywords = [
        "sqlalchemy",
        "psycopg2",
        "backend.persistence.database",
        "PostgresRuntimeSessionRepository",
        "begin_transaction"
    ]

    # Modules that are allowed to import DB/SQLAlchemy:
    # Only backend/persistence/database, backend/persistence/adapters/postgres.py and backend/persistence/service.py
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    backend_dir = os.path.join(project_dir, "backend")

    for root, dirs, files in os.walk(backend_dir):
        # Exclude database infrastructure, postgres adapters, legacy database, models, repositories, and backtest runner
        if (
            "persistence\\database" in root or "persistence/database" in root or
            "persistence\\adapters" in root or "persistence/adapters" in root or
            "\\database" in root or "/database" in root or
            "\\backtest" in root or "/backtest" in root or
            "\\models" in root or "/models" in root or
            "\\repositories" in root or "/repositories" in root
        ):
            continue
        
        for file in files:
            if not file.endswith(".py"):
                continue
            if file == "service.py" and "persistence" in root:
                # service.py is allowed to orchestrate the factories
                continue
            
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for keyword in forbidden_keywords:
                            assert keyword not in alias.name, (
                                f"Forbidden import '{alias.name}' detected in {filepath}."
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for keyword in forbidden_keywords:
                            assert keyword not in node.module, (
                                f"Forbidden import module '{node.module}' detected in {filepath}."
                            )
                        for alias in node.names:
                            assert keyword not in alias.name, (
                                f"Forbidden import name '{alias.name}' from '{node.module}' in {filepath}."
                            )
