from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntegratedRuntimeStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    STARTING = "STARTING"
    WARMING_UP = "WARMING_UP"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class CycleTriggerType(str, Enum):
    MARKET_UPDATE = "MARKET_UPDATE"
    CANDLE_CLOSE = "CANDLE_CLOSE"
    MANUAL = "MANUAL"
    REPLAY_STEP = "REPLAY_STEP"


class ComponentHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass(frozen=True)
class ComponentHealth:
    component_name: str
    status: ComponentHealthStatus
    message: str
    checked_at: str  # ISO UTC
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegratedRuntimeSnapshot:
    session_id: str
    status: IntegratedRuntimeStatus
    cycle_count: int
    successful_cycles: int
    rejected_cycles: int
    failed_cycles: int
    warmup_cycles: int
    active_symbols: List[str]
    started_at: str  # ISO UTC
    updated_at: str  # ISO UTC
    portfolio_equity: float
    available_balance: float
    open_positions: List[Dict[str, Any]]  # Concise representation of positions
    last_cycle_id: Optional[str]
    last_cycle_status: Optional[str]
    component_health: Dict[str, ComponentHealth]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    started_at: str  # ISO UTC
    completed_at: str  # ISO UTC
    total_cycles: int
    completed_cycles: int
    rejected_cycles: int
    failed_cycles: int
    warmup_cycles: int
    executions: int
    fills: int
    positions_opened: int
    positions_closed: int
    initial_equity: float
    final_equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    max_drawdown: float
    runtime_latency_ms: float
    stop_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
