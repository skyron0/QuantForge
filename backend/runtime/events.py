from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TradingEvent:
    """Base class for all immutable trading runtime events."""
    event_id: str
    event_type: str
    timestamp: str
    runtime_id: str
    session_id: str
    cycle_id: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeStarted(TradingEvent):
    pass


@dataclass(frozen=True)
class RuntimePaused(TradingEvent):
    pass


@dataclass(frozen=True)
class RuntimeResumed(TradingEvent):
    pass


@dataclass(frozen=True)
class RuntimeStopped(TradingEvent):
    pass


@dataclass(frozen=True)
class RuntimeFailed(TradingEvent):
    pass


@dataclass(frozen=True)
class RuntimeStateChanged(TradingEvent):
    old_state: str = ""
    new_state: str = ""


@dataclass(frozen=True)
class TradingCycleStarted(TradingEvent):
    pass


@dataclass(frozen=True)
class TradingCycleFinished(TradingEvent):
    pass


@dataclass(frozen=True)
class TradingCycleFailed(TradingEvent):
    pass


@dataclass(frozen=True)
class DecisionCreated(TradingEvent):
    pass


@dataclass(frozen=True)
class ProposalGenerated(TradingEvent):
    pass


@dataclass(frozen=True)
class ProposalRejected(TradingEvent):
    pass


@dataclass(frozen=True)
class RiskApproved(TradingEvent):
    pass


@dataclass(frozen=True)
class RiskRejected(TradingEvent):
    pass


@dataclass(frozen=True)
class PositionSized(TradingEvent):
    pass


@dataclass(frozen=True)
class ExecutionAuthorized(TradingEvent):
    pass


@dataclass(frozen=True)
class ExecutionRejected(TradingEvent):
    pass


@dataclass(frozen=True)
class OrderExecuted(TradingEvent):
    pass


@dataclass(frozen=True)
class PortfolioUpdated(TradingEvent):
    pass


@dataclass(frozen=True)
class PositionOpened(TradingEvent):
    pass


@dataclass(frozen=True)
class PositionClosed(TradingEvent):
    pass


@dataclass(frozen=True)
class ProtectiveTriggerActivated(TradingEvent):
    pass


@dataclass(frozen=True)
class ProtectiveExitExecuted(TradingEvent):
    pass


@dataclass(frozen=True)
class TelemetryRecorded(TradingEvent):
    pass


@dataclass(frozen=True)
class RuntimeError(TradingEvent):
    error_msg: str = ""

