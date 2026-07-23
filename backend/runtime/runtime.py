import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

from backend.orchestration.models import TradingCycleInput, TradingCycleResult, TradingCycleStatus
from backend.orchestration.orchestrator import TradingCycleOrchestrator
from backend.runtime.exceptions import InvalidStateTransitionError, RuntimeException
from backend.runtime.policy import RuntimePolicy
from backend.runtime.telemetry import RuntimeTelemetry
from backend.runtime.session import TradingSession
from backend.runtime.event_bus import BaseEventBus
from backend.runtime.scheduler import BaseScheduler, TradingCycleScheduler
from backend.runtime import events


class TradingRuntime:
    """
    Central operating system layer coordinating events, state machine, and scheduling
    across the entire QuantForge ecosystem. Remains provider-agnostic.
    """
    def __init__(
        self,
        policy: RuntimePolicy,
        event_bus: BaseEventBus,
        telemetry: RuntimeTelemetry,
        session: TradingSession,
        orchestrator: TradingCycleOrchestrator,
        scheduler_factory: Optional[Callable[..., BaseScheduler]] = None
    ) -> None:
        self.policy = policy
        self.event_bus = event_bus
        self.telemetry = telemetry
        self.session = session
        self.orchestrator = orchestrator
        
        self.runtime_id = str(uuid.uuid4())
        self._state = "INITIALIZED"

        # Construct Scheduler using factory or default implementation
        if scheduler_factory:
            self.scheduler = scheduler_factory(
                interval_seconds=policy.scheduler_interval_seconds,
                input_provider=self._provide_cycle_input,
                cycle_executor=self.run_cycle_directly,
                telemetry=telemetry
            )
        else:
            self.scheduler = TradingCycleScheduler(
                interval_seconds=policy.scheduler_interval_seconds,
                input_provider=self._provide_cycle_input,
                cycle_executor=self.run_cycle_directly,
                telemetry=telemetry
            )

        self._cycle_input_override: Optional[TradingCycleInput] = None
        self.telemetry.transition_state(self._state, datetime.now(timezone.utc).isoformat())

    @property
    def state(self) -> str:
        return self._state

    def start(self) -> None:
        """Transitions the runtime state to RUNNING and activates polling scheduler."""
        if self._state not in ("INITIALIZED", "STOPPED"):
            raise InvalidStateTransitionError(
                f"Cannot start runtime from state: {self._state}"
            )

        self._transition("STARTING")
        try:
            self.scheduler.start()
            self._transition("RUNNING")
            
            # Publish start messages
            self.event_bus.publish(
                events.RuntimeStarted(
                    event_id=str(uuid.uuid4()),
                    event_type="RuntimeStarted",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=None
                )
            )
        except Exception as e:
            self._transition("FAILED")
            self.telemetry.record_runtime_error(f"Failed to start runtime: {str(e)}")
            self.event_bus.publish(
                events.RuntimeFailed(
                    event_id=str(uuid.uuid4()),
                    event_type="RuntimeFailed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=None,
                    metadata={"error": str(e)}
                )
            )
            raise RuntimeException(f"Failed to start runtime: {str(e)}") from e

    def stop(self) -> None:
        """Transitions runtime state to STOPPING and deactivates polling scheduler."""
        if self._state not in ("RUNNING", "PAUSED"):
            raise InvalidStateTransitionError(
                f"Cannot stop runtime from state: {self._state}"
            )

        self._transition("STOPPING")
        try:
            self.scheduler.stop()
            self._transition("STOPPED")
            
            self.event_bus.publish(
                events.RuntimeStopped(
                    event_id=str(uuid.uuid4()),
                    event_type="RuntimeStopped",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=None
                )
            )
        except Exception as e:
            self._transition("FAILED")
            self.telemetry.record_runtime_error(f"Failed to stop runtime: {str(e)}")
            raise RuntimeException(f"Failed to stop runtime: {str(e)}") from e

    def pause(self) -> None:
        """Pauses execution iterations of the scheduler."""
        if self._state != "RUNNING":
            raise InvalidStateTransitionError(
                f"Cannot pause runtime from state: {self._state}"
            )

        self.scheduler.pause()
        self._transition("PAUSED")
        
        self.event_bus.publish(
            events.RuntimePaused(
                event_id=str(uuid.uuid4()),
                event_type="RuntimePaused",
                timestamp=datetime.now(timezone.utc).isoformat(),
                runtime_id=self.runtime_id,
                session_id=self.session.session_id,
                cycle_id=None
            )
        )

    def resume(self) -> None:
        """Resumes execution iterations of the scheduler."""
        if self._state != "PAUSED":
            raise InvalidStateTransitionError(
                f"Cannot resume runtime from state: {self._state}"
            )

        self.scheduler.resume()
        self._transition("RUNNING")
        
        self.event_bus.publish(
            events.RuntimeResumed(
                event_id=str(uuid.uuid4()),
                event_type="RuntimeResumed",
                timestamp=datetime.now(timezone.utc).isoformat(),
                runtime_id=self.runtime_id,
                session_id=self.session.session_id,
                cycle_id=None
            )
        )

    def shutdown(self) -> None:
        """Stops the scheduler, clears event bus bindings, and transitions to STOPPED."""
        if self._state not in ("INITIALIZED", "RUNNING", "PAUSED", "STOPPED", "FAILED"):
            raise InvalidStateTransitionError(
                f"Cannot shutdown runtime from state: {self._state}"
            )

        # Attempt stopped scheduler
        try:
            self.scheduler.stop()
        except Exception:
            pass

        self._transition("STOPPED")
        
        self.event_bus.publish(
            events.RuntimeStopped(
                event_id=str(uuid.uuid4()),
                event_type="RuntimeStopped",
                timestamp=datetime.now(timezone.utc).isoformat(),
                runtime_id=self.runtime_id,
                session_id=self.session.session_id,
                cycle_id=None,
                metadata={"shutdown": True}
            )
        )
        
        # Clear EventBus subscriptions safely to clean up handlers
        self.event_bus.clear()

    def set_cycle_input(self, input_data: TradingCycleInput) -> None:
        """Injects or overrides cycle inputs (mostly used for deterministic test injection)."""
        self._cycle_input_override = input_data

    def run_cycle_directly(self, input_data: TradingCycleInput) -> TradingCycleResult:
        """
        Executes a cycle directly with the Orchestrator,
        incrementing session cycles and publishing event bus progression logs.
        """
        cycle_num = self.session.increment_cycle()
        cycle_id = f"cycle-{self.session.session_id}-{cycle_num}"
        
        time_now = datetime.now(timezone.utc).isoformat()
        
        self.event_bus.publish(
            events.TradingCycleStarted(
                event_id=str(uuid.uuid4()),
                event_type="TradingCycleStarted",
                timestamp=time_now,
                runtime_id=self.runtime_id,
                session_id=self.session.session_id,
                cycle_id=cycle_id
            )
        )

        try:
            res = self.orchestrator.run_cycle(input_data)
            
            # Publish progression event messages
            self._publish_cycle_progress(cycle_id, res)
            
            if res.status == TradingCycleStatus.FAILED:
                self.event_bus.publish(
                    events.TradingCycleFailed(
                        event_id=str(uuid.uuid4()),
                        event_type="TradingCycleFailed",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        runtime_id=self.runtime_id,
                        session_id=self.session.session_id,
                        cycle_id=cycle_id,
                        metadata={"stage": res.rejection_stage, "reason": res.rejection_reason}
                    )
                )
            else:
                self.event_bus.publish(
                    events.TradingCycleFinished(
                        event_id=str(uuid.uuid4()),
                        event_type="TradingCycleFinished",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        runtime_id=self.runtime_id,
                        session_id=self.session.session_id,
                        cycle_id=cycle_id,
                        metadata={"status": res.status.value}
                    )
                )
            return res
        except Exception as e:
            self.telemetry.record_runtime_error(f"Critical cycle error: {str(e)}")
            self.event_bus.publish(
                events.RuntimeError(
                    event_id=str(uuid.uuid4()),
                    event_type="RuntimeError",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    error_msg=f"Critical cycle execution crash: {str(e)}"
                )
            )
            self.event_bus.publish(
                events.TradingCycleFailed(
                    event_id=str(uuid.uuid4()),
                    event_type="TradingCycleFailed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"error": str(e)}
                )
            )
            raise

    def _provide_cycle_input(self) -> Optional[TradingCycleInput]:
        # Polled by scheduler. Returns override if set.
        val = self._cycle_input_override
        # Clear override after consumption to require fresh setups
        self._cycle_input_override = None
        return val

    def _transition(self, new_state: str) -> None:
        old_state = self._state
        self._state = new_state
        time_now = datetime.now(timezone.utc).isoformat()
        
        self.telemetry.transition_state(new_state, time_now)
        
        # Publish mutable observing event
        self.event_bus.publish(
            events.RuntimeStateChanged(
                event_id=str(uuid.uuid4()),
                event_type="RuntimeStateChanged",
                timestamp=time_now,
                runtime_id=self.runtime_id,
                session_id=self.session.session_id,
                cycle_id=None,
                old_state=old_state,
                new_state=new_state
            )
        )

    def _publish_cycle_progress(self, cycle_id: str, res: TradingCycleResult) -> None:
        time_now = datetime.now(timezone.utc).isoformat()
        
        # 1. Decision Creation and Signal details
        if res.fusion_id:
            self.event_bus.publish(
                events.DecisionCreated(
                    event_id=str(uuid.uuid4()),
                    event_type="DecisionCreated",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"fusion_id": res.fusion_id, "intel_used": res.intelligence_used}
                )
            )

        # 2. Decision Fusion outputs
        if res.proposal_generated:
            self.event_bus.publish(
                events.ProposalGenerated(
                    event_id=str(uuid.uuid4()),
                    event_type="ProposalGenerated",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"proposal_id": res.proposal_id}
                )
            )
        elif res.status == TradingCycleStatus.FUSION_REJECTED:
            self.event_bus.publish(
                events.ProposalRejected(
                    event_id=str(uuid.uuid4()),
                    event_type="ProposalRejected",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"reason": res.rejection_reason}
                )
            )

        # 3. Risk Guard
        if res.risk_authorized:
            self.event_bus.publish(
                events.RiskApproved(
                    event_id=str(uuid.uuid4()),
                    event_type="RiskApproved",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"risk_auth_id": res.risk_authorization_id}
                )
            )
        elif res.status == TradingCycleStatus.RISK_REJECTED:
            self.event_bus.publish(
                events.RiskRejected(
                    event_id=str(uuid.uuid4()),
                    event_type="RiskRejected",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"reason": res.rejection_reason}
                )
            )

        # 4. Sizing
        if res.sizing_id:
            self.event_bus.publish(
                events.PositionSized(
                    event_id=str(uuid.uuid4()),
                    event_type="PositionSized",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"sizing_id": res.sizing_id}
                )
            )

        # 5. Execution Authorization
        if res.execution_authorized:
            self.event_bus.publish(
                events.ExecutionAuthorized(
                    event_id=str(uuid.uuid4()),
                    event_type="ExecutionAuthorized",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"execution_auth_id": res.execution_authorization_id, "intent_id": res.intent_id}
                )
            )
        elif res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED:
            self.event_bus.publish(
                events.ExecutionRejected(
                    event_id=str(uuid.uuid4()),
                    event_type="ExecutionRejected",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"reason": res.rejection_reason}
                )
            )

        # 6. Adapter Executed
        if res.executed:
            self.event_bus.publish(
                events.OrderExecuted(
                    event_id=str(uuid.uuid4()),
                    event_type="OrderExecuted",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"execution_id": res.execution_id, "fill_ids": res.fill_ids}
                )
            )
        elif res.status == TradingCycleStatus.EXECUTION_FAILED:
            self.event_bus.publish(
                events.RuntimeError(
                    event_id=str(uuid.uuid4()),
                    event_type="RuntimeError",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    error_msg=f"Execution Failed: {res.rejection_reason}"
                )
            )

        # 7. Portfolio Engine Ingestion
        if res.portfolio_updated:
            self.event_bus.publish(
                events.PortfolioUpdated(
                    event_id=str(uuid.uuid4()),
                    event_type="PortfolioUpdated",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id
                )
            )

        # 8. Lifecycle Engine Registration
        if res.lifecycle_registered:
            self.event_bus.publish(
                events.PositionOpened(
                    event_id=str(uuid.uuid4()),
                    event_type="PositionOpened",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    metadata={"symbol": res.symbol}
                )
            )
        elif res.status == TradingCycleStatus.LIFECYCLE_REGISTRATION_FAILED:
            self.event_bus.publish(
                events.RuntimeError(
                    event_id=str(uuid.uuid4()),
                    event_type="RuntimeError",
                    timestamp=time_now,
                    runtime_id=self.runtime_id,
                    session_id=self.session.session_id,
                    cycle_id=cycle_id,
                    error_msg=f"Lifecycle registration failed: {res.rejection_reason}"
                )
            )
