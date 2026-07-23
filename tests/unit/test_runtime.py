import os
import ast
import pytest
import threading
import time
import dataclasses
from datetime import datetime, timezone
from typing import Optional, List, Any
from unittest.mock import Mock

# Orchestrator modules to mock/construct cycle inputs
from backend.orchestration.models import TradingCycleInput, TradingCycleResult, TradingCycleStatus
from backend.orchestration.orchestrator import TradingCycleOrchestrator

# Runtime package imports
from backend.runtime.exceptions import (
    RuntimeException,
    InvalidStateTransitionError,
    PolicyValidationError,
    PublishError,
    SessionError,
    SchedulerError,
)
from backend.runtime.policy import RuntimePolicy
from backend.runtime.telemetry import RuntimeTelemetry
from backend.runtime.events import (
    TradingEvent,
    RuntimeStarted,
    RuntimePaused,
    RuntimeResumed,
    RuntimeStopped,
    RuntimeFailed,
    RuntimeStateChanged,
    TradingCycleStarted,
    TradingCycleFinished,
    TradingCycleFailed,
    DecisionCreated,
    ProposalGenerated,
    ProposalRejected,
    RiskApproved,
    RiskRejected,
    PositionSized,
    ExecutionAuthorized,
    ExecutionRejected,
    OrderExecuted,
    PortfolioUpdated,
    PositionOpened,
    PositionClosed,
    RuntimeError,
)
from backend.runtime.event_bus import EventBus
from backend.runtime.dispatcher import Dispatcher
from backend.runtime.session import TradingSession
from backend.runtime.scheduler import TradingCycleScheduler
from backend.runtime.runtime import TradingRuntime


@pytest.fixture
def base_policy() -> RuntimePolicy:
    return RuntimePolicy(
        scheduler_interval_seconds=0.1,
        max_runtime_duration_seconds=5.0,
        max_event_queue_size=100,
        clock_skew_tolerance_seconds=2.0,
        telemetry_enabled=True,
        max_dispatch_latency_ms=10.0,
    )


@pytest.fixture
def mock_telemetry() -> RuntimeTelemetry:
    return RuntimeTelemetry(enabled=True)


@pytest.fixture
def mock_session() -> TradingSession:
    return TradingSession(metadata={"test_env": True})


@pytest.fixture
def mock_orchestrator() -> TradingCycleOrchestrator:
    # We stub/mock the orchestrator so it doesn't touch downstream engines.
    orchestrator = Mock(spec=TradingCycleOrchestrator)
    
    # Setup dummy default response
    dummy_result = TradingCycleResult(
        cycle_id="cycle-abc-123",
        symbol="BTC/USDT",
        timeframe="1m",
        status=TradingCycleStatus.COMPLETED,
        proposal_generated=True,
        proposal_id="prop-123",
        risk_authorized=True,
        risk_authorization_id="risk-123",
        sizing_id="size-123",
        execution_authorized=True,
        execution_authorization_id="exec-auth-123",
        intent_id="intent-123",
        executed=True,
        execution_id="exec-123",
        fill_ids=["fill-1", "fill-2"],
        portfolio_updated=True,
        lifecycle_registered=True,
        rejection_stage=None,
        rejection_reason=None,
        started_at="2026-07-22T12:00:00Z",
        completed_at="2026-07-22T12:00:01Z",
        latency_ms=10.0,
        total_latency_ms=10.0,
        stage_timings={"decide": 5.0, "risk": 2.0},
        fusion_id="fusion-123",
        intelligence_used=True,
    )
    orchestrator.run_cycle.return_value = dummy_result
    return orchestrator


# 1. Event creation & immutability
def test_event_immutability():
    evt = TradingEvent(
        event_id="evt-1",
        event_type="TestEvent",
        timestamp=datetime.now(timezone.utc).isoformat(),
        runtime_id="run-1",
        session_id="sess-1",
        cycle_id=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        evt.event_type = "Modified"  # type: ignore


# 2. Publish, Subscribe, Unsubscribe
def test_event_bus_pub_sub_unsub():
    bus = EventBus(max_queue_size=10)
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("TestEvent", handler)
    
    evt = TradingEvent(
        event_id="evt-1",
        event_type="TestEvent",
        timestamp=datetime.now(timezone.utc).isoformat(),
        runtime_id="run-1",
        session_id="sess-1",
        cycle_id=None
    )
    
    bus.publish(evt)
    assert len(received) == 1
    assert received[0].event_id == "evt-1"

    bus.unsubscribe("TestEvent", handler)
    bus.publish(evt)
    assert len(received) == 1  # count unchanged


# 3. Duplicate subscribers prevention
def test_event_bus_duplicate_subscribers():
    bus = EventBus(max_queue_size=10)
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("TestEvent", handler)
    bus.subscribe("TestEvent", handler)  # duplicate subscription

    evt = TradingEvent(
        event_id="evt-1",
        event_type="TestEvent",
        timestamp="2026-07-22T00:00:00Z",
        runtime_id="run-1",
        session_id="sess-1",
        cycle_id=None
    )
    bus.publish(evt)
    # Set uniqueness checks ensure it gets executed exactly once
    assert len(received) == 1


# 4. Dispatcher Error Isolation
def test_dispatcher_error_isolation(mock_telemetry):
    dispatcher = Dispatcher(telemetry=mock_telemetry)
    bus = EventBus(max_queue_size=10, dispatcher=dispatcher)
    
    calls = []

    def failing_handler(evt):
        raise ValueError("Simulated subscriber crash")

    def succeeding_handler(evt):
        calls.append(evt)

    bus.subscribe("TestEvent", failing_handler)
    bus.subscribe("TestEvent", succeeding_handler)

    evt = TradingEvent(
        event_id="evt-1",
        event_type="TestEvent",
        timestamp="2026-07-22T00:00:00Z",
        runtime_id="run-1",
        session_id="sess-1",
        cycle_id=None
    )
    
    bus.publish(evt)
    
    # The succeeding handler must still run
    assert len(calls) == 1
    # Telemetry should record the failures
    assert mock_telemetry.failed_handlers == 1
    assert mock_telemetry.failed_events == 1
    assert len(mock_telemetry.runtime_errors) == 1
    assert "Simulated subscriber crash" in mock_telemetry.runtime_errors[0]


# 5. Dispatch Ordering (nested publish execution order)
def test_dispatch_nesting_order_determinism():
    bus = EventBus(max_queue_size=10)
    execution_order = []

    def handler_first(evt):
        execution_order.append("A_started")
        nested_evt = TradingEvent(
            event_id="evt-nested",
            event_type="NestedEvent",
            timestamp="2026-07-22T00:00:00",
            runtime_id="run-1",
            session_id="sess-1",
            cycle_id=None
        )
        bus.publish(nested_evt)
        execution_order.append("A_finished")

    def handler_nested(evt):
        execution_order.append("B_executed")

    bus.subscribe("OuterEvent", handler_first)
    bus.subscribe("NestedEvent", handler_nested)

    outer_evt = TradingEvent(
        event_id="evt-outer",
        event_type="OuterEvent",
        timestamp="2026-07-22T00:00:00",
        runtime_id="run-1",
        session_id="sess-1",
        cycle_id=None
    )
    bus.publish(outer_evt)

    # Queue-based loop ensures outer finishes COMPLETELY before inner begins
    assert execution_order == ["A_started", "A_finished", "B_executed"]


# 6. EventBus Queue Overflow
def test_event_bus_queue_overflow():
    # If the queue runs out of space, PublishError must be raised
    bus = EventBus(max_queue_size=1)
    
    # We block deep execution inside a handler and publish nested
    def recursive_handler(evt):
        nested_1 = TradingEvent("2", "OtherEvent", "ts", "r", "s", None)
        nested_2 = TradingEvent("3", "OtherEvent", "ts", "r", "s", None)
        # Nested 1 enqueues successfully (max_queue_size=1)
        bus.publish(nested_1)
        # Enqueuing nested 2 causes overflow
        with pytest.raises(PublishError):
            bus.publish(nested_2)

    bus.subscribe("TestEvent", recursive_handler)
    initial_evt = TradingEvent("1", "TestEvent", "ts", "r", "s", None)
    bus.publish(initial_evt)


# 7. Telemetry Averages Latency Calculations
def test_telemetry_latency_tracking(mock_telemetry):
    mock_telemetry.record_cycle(10.0)
    mock_telemetry.record_cycle(20.0)
    mock_telemetry.record_dispatch(2.0)
    mock_telemetry.record_dispatch(4.0)

    assert mock_telemetry.cycle_count == 2
    assert mock_telemetry.average_cycle_latency_ms == 15.0
    assert mock_telemetry.average_dispatch_latency_ms == 3.0


# 8. Runtime Policy Validation
def test_runtime_policy_validation():
    # Invalid negative parameter raises PolicyValidationError
    with pytest.raises(PolicyValidationError):
        RuntimePolicy(scheduler_interval_seconds=-1.0)
    with pytest.raises(PolicyValidationError):
        RuntimePolicy(max_event_queue_size=0)
    with pytest.raises(PolicyValidationError):
        RuntimePolicy(max_dispatch_latency_ms=-5.0)


# 9. Mutable Trading Session synchronization check
def test_session_state_updates():
    sess = TradingSession(session_id="test-sess", metadata={"mode": "paper"})
    assert sess.session_id == "test-sess"
    assert sess.cycle_counter == 0

    sess.increment_cycle()
    assert sess.cycle_counter == 1

    sess.update_metadata("key1", "val1")
    assert sess.get_metadata("key1") == "val1"


# 10. Manual Scheduler ticking
def test_manual_scheduler_tick(mock_orchestrator, mock_telemetry):
    inputs = []
    
    def input_provider():
        if not inputs:
            return None
        return inputs.pop(0)

    # Manual deterministic Scheduler
    scheduler = TradingCycleScheduler(
        interval_seconds=1.0,
        input_provider=input_provider,
        cycle_executor=mock_orchestrator.run_cycle,
        telemetry=mock_telemetry
    )
    
    # We do NOT run scheduler.start() (which spawns a thread)
    # Instead, we test it purely synchronously via manual tick hooks
    mock_input = TradingCycleInput(
        ml_signal=Mock(),
        risk_context=Mock(),
        position_sizing_context=Mock(),
        execution_context=Mock(),
        paper_execution_context=Mock(),
        timestamp="2026-07-22T12:00:00Z"
    )
    inputs.append(mock_input)

    scheduler.tick()  # scheduler is not started, so it should exit early
    assert mock_telemetry.scheduler_iterations == 0

    # Start it temporarily to allow ticks (will block polling loop if start runs, but we can set internal flag instead)
    # Actually, let's start the Scheduler, but override running state/paused flags manually or test it started:
    scheduler._running = True
    scheduler.tick()
    assert mock_telemetry.scheduler_iterations == 1
    assert mock_orchestrator.run_cycle.call_count == 1


# 11. State Machine Validity transitions
def test_runtime_lifecycle_transitions(base_policy, mock_telemetry, mock_session, mock_orchestrator):
    bus = EventBus(max_queue_size=100)
    runtime = TradingRuntime(
        policy=base_policy,
        event_bus=bus,
        telemetry=mock_telemetry,
        session=mock_session,
        orchestrator=mock_orchestrator
    )
    
    assert runtime.state == "INITIALIZED"
    
    # State changes trace list
    state_updates = []
    bus.subscribe("RuntimeStateChanged", lambda evt: state_updates.append((evt.old_state, evt.new_state)))

    runtime.start()
    assert runtime.state == "RUNNING"
    
    runtime.pause()
    assert runtime.state == "PAUSED"
    
    runtime.resume()
    assert runtime.state == "RUNNING"
    
    runtime.stop()
    assert runtime.state == "STOPPED"

    # Verify transition logs
    assert ("INITIALIZED", "STARTING") in state_updates
    assert ("STARTING", "RUNNING") in state_updates
    assert ("RUNNING", "PAUSED") in state_updates
    assert ("PAUSED", "RUNNING") in state_updates
    assert ("RUNNING", "STOPPING") in state_updates
    assert ("STOPPING", "STOPPED") in state_updates

    # Invalid transitions fail-closed
    with pytest.raises(InvalidStateTransitionError):
        runtime.resume()  # cannot resume stopped


# 12. End-to-end event progression and runtime execution updates
def test_runtime_cycle_event_propagation(base_policy, mock_telemetry, mock_session, mock_orchestrator):
    bus = EventBus(max_queue_size=100)
    runtime = TradingRuntime(
        policy=base_policy,
        event_bus=bus,
        telemetry=mock_telemetry,
        session=mock_session,
        orchestrator=mock_orchestrator
    )
    
    events_triggered = []
    def track_events(evt):
        events_triggered.append(evt.event_type)

    event_types = [
        "TradingCycleStarted", "TradingCycleFinished", "DecisionCreated", "ProposalGenerated",
        "RiskApproved", "PositionSized", "ExecutionAuthorized", "OrderExecuted", "PortfolioUpdated",
        "PositionOpened"
    ]
    for et in event_types:
        bus.subscribe(et, track_events)
        
    mock_input = TradingCycleInput(
        ml_signal=Mock(),
        risk_context=Mock(),
        position_sizing_context=Mock(),
        execution_context=Mock(),
        paper_execution_context=Mock(),
        timestamp="2026-07-22T12:00:00Z"
    )
    runtime.run_cycle_directly(mock_input)

    # All stage events should fire in order
    assert "TradingCycleStarted" in events_triggered
    assert "DecisionCreated" in events_triggered
    assert "ProposalGenerated" in events_triggered
    assert "RiskApproved" in events_triggered
    assert "PositionSized" in events_triggered
    assert "ExecutionAuthorized" in events_triggered
    assert "OrderExecuted" in events_triggered
    assert "PortfolioUpdated" in events_triggered
    assert "PositionOpened" in events_triggered
    assert "TradingCycleFinished" in events_triggered


# 13. System Import dependency Isolation rules validation
def test_runtime_package_execution_isolation():
    forbidden_keywords = [
        "ccxt",
        "binance",
        "bybit",
        "brokers",
        "exchanges",
        "ReasoningEngine",
        "OllamaProvider",
        "ollama",
    ]

    runtime_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    runtime_dir = os.path.join(runtime_dir, "backend", "runtime")

    python_files = [
        os.path.join(runtime_dir, f)
        for f in os.listdir(runtime_dir)
        if f.endswith(".py")
    ]

    for filepath in python_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # AST analysis
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
