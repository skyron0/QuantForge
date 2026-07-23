import uuid
import pytest
from decimal import Decimal
from sqlalchemy import create_engine
from backend.persistence.policy import PersistencePolicy
from backend.persistence.telemetry import PersistenceTelemetry
from backend.persistence.service import PersistenceService
from backend.persistence.database.connection import db_session, get_engine_url
from backend.persistence.database.migrations import run_migrations
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord, DecisionRecord,
    TradeProposalRecord, RiskAuthorizationRecord, PositionSizingRecord,
    OrderIntentRecord, ExecutionRecord, FillRecord,
    PortfolioSnapshotRecord, PositionSnapshotRecord, PositionLifecycleRecord, AuditEventRecord
)

# Detect if PostgreSQL is active and accessible
db_url = get_engine_url()
db_available = False
try:
    engine = create_engine(db_url, connect_args={"connect_timeout": 2})
    with engine.connect() as conn:
        db_available = True
except Exception:
    db_available = False

# Auto-skip all tests in this file if Postgres is down
pytestmark = pytest.mark.skipif(not db_available, reason="PostgreSQL is not running or accessible")


@pytest.fixture(scope="module", autouse=True)
def setup_database_schema():
    # Make sure migrations run cleanly before any integration test
    run_migrations()


@pytest.fixture
def postgres_policy() -> PersistencePolicy:
    return PersistencePolicy(
        persistence_enabled=True,
        persistence_backend="postgres",
        database_pool_size=5,
        database_pool_overflow=10,
        database_timeout_seconds=2.0
    )


@pytest.fixture
def telemetry() -> PersistenceTelemetry:
    return PersistenceTelemetry(enabled=True)


@pytest.fixture
def postgres_service(postgres_policy, telemetry) -> PersistenceService:
    return PersistenceService(policy=postgres_policy, telemetry=telemetry)


def test_postgres_runtime_session_repo(postgres_service):
    session_id = f"sess-pg-test-{uuid.uuid4()}"
    started_at = "2026-07-22T12:00:00+00:00"
    
    record = RuntimeSessionRecord(
        session_id=session_id,
        status="STARTED",
        started_at=started_at,
        metadata={"env": "integration-test"}
    )
    
    postgres_service.save_session(record)
    
    # Retrieve and verify
    fetched = postgres_service.get_session(session_id)
    assert fetched is not None
    assert fetched.session_id == session_id
    assert fetched.status == "STARTED"
    assert fetched.started_at == started_at
    assert fetched.metadata.get("env") == "integration-test"

    # Stop session
    stopped_record = RuntimeSessionRecord(
        session_id=session_id,
        status="STOPPED",
        started_at=started_at,
        stopped_at="2026-07-22T12:15:00+00:00",
        metadata={"env": "integration-test"}
    )
    postgres_service.save_session(stopped_record)
    
    fetched = postgres_service.get_session(session_id)
    assert fetched is not None
    assert fetched.status == "STOPPED"
    assert fetched.stopped_at == "2026-07-22T12:15:00+00:00"


def test_postgres_trading_cycle_repo(postgres_service):
    session_id = f"sess-pg-cycle-{uuid.uuid4()}"
    cycle_id = f"cycle-pg-test-{uuid.uuid4()}"
    
    # Prerequisite session
    postgres_service.save_session(
        RuntimeSessionRecord(session_id=session_id, status="STARTED", started_at="2026-07-22T12:00:00Z")
    )
    
    record = TradingCycleRecord(
        cycle_id=cycle_id,
        session_id=session_id,
        cycle_index=1,
        status="COMPLETED",
        started_at="2026-07-22T12:00:00+00:00",
        completed_at="2026-07-22T12:00:01+00:00",
        latency_ms=15.5,
        total_latency_ms=16.0,
        fusion_id="fusion-1",
        proposal_id="proposal-1",
        risk_authorized=True,
        execution_authorized=True,
        executed=True,
        policy_version="policy-v1",
        metadata={"test": 12}
    )
    
    postgres_service.save_trading_cycle(record)
    
    fetched = postgres_service.get_trading_cycle(cycle_id)
    assert fetched is not None
    assert fetched.cycle_id == cycle_id
    assert fetched.session_id == session_id
    assert fetched.status == "COMPLETED"
    assert fetched.latency_ms == 15.5
    assert fetched.risk_authorized is True
    assert fetched.metadata.get("test") == 12


def test_postgres_execution_and_fills_atomic_precisions(postgres_service):
    session_id = f"sess-pg-exec-{uuid.uuid4()}"
    cycle_id = f"cycle-pg-exec-{uuid.uuid4()}"
    exec_id = f"exec-pg-{uuid.uuid4()}"
    intent_id = f"intent-pg-{uuid.uuid4()}"
    proposal_id = f"prop-pg-{uuid.uuid4()}"
    auth_id = f"auth-pg-{uuid.uuid4()}"
    sizing_id = f"sizing-pg-{uuid.uuid4()}"
    
    # Save prerequisite session and cycle
    postgres_service.save_session(
        RuntimeSessionRecord(session_id=session_id, status="STARTED", started_at="2026-07-22T12:00:00Z")
    )
    postgres_service.save_trading_cycle(
        TradingCycleRecord(
            cycle_id=cycle_id, session_id=session_id, cycle_index=1, status="COMPLETED",
            started_at="2026-07-22T12:00:00Z", completed_at="2026-07-22T12:00:01Z",
            latency_ms=10, total_latency_ms=10
        )
    )
    
    # Save other domain models to postgres using service (to check schema mapping works for all entity types)
    postgres_service.save_decision(
        DecisionRecord(
            decision_id=proposal_id, cycle_id=cycle_id, symbol="BTC/USDT", timeframe="1m",
            direction="BUY", confidence=0.85, fusion_score=0.9, agreement_score=1.0,
            ml_contribution=0.5, intelligence_contribution=0.5, intelligence_used=True,
            intelligence_age_seconds=5.0, policy_version="v1", source_model_version="sm-1",
            reasoning_request_id=None, risk_flags=[], timestamp="2026-07-22T12:00:00Z"
        )
    )
    postgres_service.save_trade_proposal(
        TradeProposalRecord(
            proposal_id=proposal_id, decision_id=proposal_id, symbol="BTC/USDT",
            direction="BUY", confidence=0.85, fusion_score=0.9, source_model_version="sm-1",
            fusion_policy_version="v1", reasoning_request_id=None, created_at="2026-07-22T12:00:00Z",
            expires_at="2026-07-22T12:01:00Z", risk_flags=[]
        )
    )
    postgres_service.save_risk_authorization(
        RiskAuthorizationRecord(
            authorization_id=auth_id, proposal_id=proposal_id, symbol="BTC/USDT",
            direction="BUY", status="APPROVED", original_confidence=0.85, effective_confidence=0.85,
            rejection_reasons=[], adjustment_reasons=[], triggered_rules=[], policy_version="v1",
            source_model_version="sm-1", fusion_policy_version="v1", proposal_created_at="2026-07-22T12:00:00Z",
            evaluated_at="2026-07-22T12:00:01Z", latency_ms=1.5, requested_risk_fraction=0.01,
            authorized_risk_fraction=0.01
        )
    )
    postgres_service.save_position_sizing(
        PositionSizingRecord(
            sizing_id=sizing_id, proposal_id=proposal_id, symbol="BTC/USDT", direction="BUY",
            quantity=Decimal("0.02456789"), position_notional=Decimal("1500.25"), entry_price=Decimal("61073.5"),
            stop_loss_price=Decimal("60000"), stop_distance_absolute=Decimal("1073.5"), stop_distance_fraction=Decimal("0.017"),
            authorized_risk_fraction=Decimal("0.01"), risk_amount=Decimal("15"), leverage=Decimal("1"),
            estimated_margin_required=Decimal("1500.25"), policy_version="v1", created_at="2026-07-22T12:00:00Z",
            authorization_id=auth_id, source_model_version="sm-1"
        )
    )
    postgres_service.save_order_intent(
        OrderIntentRecord(
            intent_id=intent_id, idempotency_key="idemp-pg-1", proposal_id=proposal_id,
            risk_authorization_id=auth_id, sizing_id=sizing_id, symbol="BTC/USDT",
            direction="BUY", quantity=Decimal("0.02456789"), order_type="LIMIT",
            limit_price=Decimal("61073.5"), stop_loss=Decimal("60000"), take_profit=Decimal("63000"),
            environment="SHADOW", source_model_version="sm-1", fusion_policy_version="v1",
            risk_policy_version="v1", position_sizing_policy_version="v1", execution_policy_version="v1",
            reasoning_request_id=None, created_at="2026-07-22T12:00:00Z", expires_at="2026-07-22T12:05:00Z"
        )
    )

    # Save Execution and nested Fills
    exec_rec = ExecutionRecord(
        execution_id=exec_id,
        intent_id=intent_id,
        proposal_id=proposal_id,
        risk_authorization_id=auth_id,
        sizing_id=sizing_id,
        symbol="BTC/USDT",
        direction="BUY",
        requested_quantity=Decimal("0.02456789"),
        filled_quantity=Decimal("0.02456780"),
        average_fill_price=Decimal("61073.45678901"),
        total_notional=Decimal("1500.24567890"),
        total_fees=Decimal("0.75234567"),
        total_slippage=Decimal("0.12345678"),
        status="FILLED",
        rejection_reason="",
        adapter_name="paper",
        environment="SHADOW",
        started_at="2026-07-22T12:00:00Z",
        completed_at="2026-07-22T12:00:01Z",
        latency_ms=120.0,
        policy_version="v1"
    )
    
    fill_id_1 = f"fill1-{uuid.uuid4()}"
    fill_id_2 = f"fill2-{uuid.uuid4()}"
    fills = [
        FillRecord(
            fill_id=fill_id_1, execution_id=exec_id, intent_id=intent_id, symbol="BTC/USDT",
            direction="BUY", quantity=Decimal("0.01000000"), price=Decimal("61073.40000000"),
            notional=Decimal("610.73400000"), fee=Decimal("0.30536700"), slippage_amount=Decimal("0.05000000"),
            timestamp="2026-07-22T12:00:00Z"
        ),
        FillRecord(
            fill_id=fill_id_2, execution_id=exec_id, intent_id=intent_id, symbol="BTC/USDT",
            direction="BUY", quantity=Decimal("0.01456780"), price=Decimal("61073.49581886"),
            notional=Decimal("889.51167890"), fee=Decimal("0.44697867"), slippage_amount=Decimal("0.07345678"),
            timestamp="2026-07-22T12:00:01Z"
        )
    ]
    
    postgres_service.save_execution(exec_rec, fills)

    # Validate retrieved fields maintain exact precision
    with db_session() as session:
        # Search the PostgreSQL repositories direct
        repo = postgres_service.execution_repository(session)
        fetched_exec = repo.get_by_id(exec_id)
        assert fetched_exec is not None
        assert fetched_exec.requested_quantity == Decimal("0.02456789")
        assert fetched_exec.average_fill_price == Decimal("61073.45678901")
        assert fetched_exec.total_slippage == Decimal("0.12345678")
        
        # Test fill retrieval
        fetched_fills = repo.get_fills_for_execution(exec_id)
        assert len(fetched_fills) == 2
        assert fetched_fills[0].fill_id in (fill_id_1, fill_id_2)
        assert fetched_fills[0].quantity in (Decimal("0.01000000"), Decimal("0.01456780"))


def test_postgres_portfolio_snapshot_and_lifecycle(postgres_service):
    portfolio_id = f"port-pg-{uuid.uuid4()}"
    snapshot_id = f"snap-pg-{uuid.uuid4()}"
    lifecycle_id = f"lifec-pg-{uuid.uuid4()}"
    pos_id = f"pos-pg-{uuid.uuid4()}"
    
    # Save a portfolio snapshot
    snapshot = PortfolioSnapshotRecord(
        portfolio_snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
        initial_balance=Decimal("100000.00"),
        cash_balance=Decimal("98500.00"),
        equity=Decimal("100000.25"),
        realized_pnl=Decimal("-10.00"),
        unrealized_pnl=Decimal("10.25"),
        total_fees=Decimal("2.50"),
        used_margin=Decimal("1500.25"),
        available_balance=Decimal("96999.75"),
        gross_exposure=Decimal("1500.25"),
        net_exposure=Decimal("1500.25"),
        open_position_count=1,
        timestamp="2026-07-22T12:00:00Z",
        positions=[
            PositionSnapshotRecord(
                position_id=pos_id,
                portfolio_snapshot_id=snapshot_id,
                symbol="BTC/USDT",
                side="LONG",
                quantity=Decimal("0.02456789"),
                average_entry_price=Decimal("61073.45678901"),
                current_price=Decimal("61073.5"),
                position_notional=Decimal("1500.25"),
                unrealized_pnl=Decimal("10.25"),
                realized_pnl=Decimal("0.00"),
                accumulated_fees=Decimal("0.75"),
                leverage=Decimal("1.00"),
                margin_used=Decimal("1500.25"),
                opened_at="2026-07-22T12:00:00Z",
                updated_at="2026-07-22T12:00:01Z"
            )
        ]
    )
    
    postgres_service.save_portfolio_snapshot(snapshot)
    
    # Retrieve and verify snapshot
    with db_session() as session:
        repo = postgres_service.portfolio_repository(session)
        fetched_snap = repo.get_latest_snapshot(portfolio_id)
        assert fetched_snap is not None
        assert fetched_snap.portfolio_snapshot_id == snapshot_id
        assert fetched_snap.equity == Decimal("100000.25")
        assert len(fetched_snap.positions) == 1
        assert fetched_snap.positions[0].position_id == pos_id
        assert fetched_snap.positions[0].quantity == Decimal("0.02456789")

    # Save Position Lifecycle
    lifecycle = PositionLifecycleRecord(
        lifecycle_id=lifecycle_id,
        position_id=pos_id,
        symbol="BTC/USDT",
        side="LONG",
        quantity=Decimal("0.02456789"),
        average_entry_price=Decimal("61073.45678901"),
        stop_loss=Decimal("60000.00"),
        take_profit=Decimal("63000.00"),
        trailing_stop_enabled=True,
        trailing_distance=Decimal("500.00"),
        trailing_activation_price=Decimal("61500.00"),
        highest_price_since_entry=Decimal("61250.00"),
        lowest_price_since_entry=Decimal("61050.00"),
        active_trailing_stop_price=Decimal("60500.00"),
        status="OPEN",
        created_at="2026-07-22T12:00:00Z",
        updated_at="2026-07-22T12:00:01Z",
        policy_version="lifec-v1"
    )
    postgres_service.save_position_lifecycle(lifecycle)
    
    fetched_lf = postgres_service.get_position_lifecycle(lifecycle_id)
    assert fetched_lf is not None
    assert fetched_lf.lifecycle_id == lifecycle_id
    assert fetched_lf.status == "OPEN"
    assert fetched_lf.active_trailing_stop_price == Decimal("60500.00")


def test_postgres_audit_ledger_continuity(postgres_service):
    session_id = f"sess-pg-audit-{uuid.uuid4()}"
    
    # Event 1
    evt1 = AuditEventRecord(
        audit_id=str(uuid.uuid4()), session_id=session_id, cycle_id=None,
        event_type="RuntimeStarted", entity_type="session", entity_id=session_id,
        symbol=None, timestamp="2026-07-22T12:00:00Z", source_component="runtime",
        status="SUCCESS", payload={}
    )
    postgres_service.save_audit_event(evt1)
    
    # Pick up first hash
    chain = postgres_service.get_audit_trail(session_id)
    assert len(chain) == 1
    h1 = chain[0].hash
    
    # Event 2 correctly chained
    evt2 = AuditEventRecord(
        audit_id=str(uuid.uuid4()), session_id=session_id, cycle_id=None,
        event_type="RuntimeStopped", entity_type="session", entity_id=session_id,
        symbol=None, timestamp="2026-07-22T12:01:00Z", source_component="runtime",
        status="SUCCESS", payload={}, previous_hash=h1
    )
    postgres_service.save_audit_event(evt2)
    
    chain = postgres_service.get_audit_trail(session_id)
    assert len(chain) == 2
    assert chain[1].previous_hash == h1
