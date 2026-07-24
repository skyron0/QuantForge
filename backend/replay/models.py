from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime

class ReplayStatus(Enum):
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"

@dataclass
class ReplayDatasetMetadata:
    dataset_hash: str
    row_count: int
    symbols: List[str]
    start_time: str
    end_time: str

@dataclass
class ReplaySessionConfig:
    initial_capital: float
    max_cycles: int
    seed: int
    dataset_path: str
    enabled_symbols: List[str]
    timeframe: str = "1m"

@dataclass
class ReplayProgress:
    total_steps: int
    processed_steps: int
    current_time: str
    percent_complete: float

@dataclass
class ReplaySessionResult:
    session_id: str
    status: ReplayStatus
    config: ReplaySessionConfig
    dataset_metadata: ReplayDatasetMetadata
    progress: ReplayProgress
    initial_equity: float
    final_equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    determinism_hash: str
    error_message: Optional[str] = None
    portfolio_history: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
