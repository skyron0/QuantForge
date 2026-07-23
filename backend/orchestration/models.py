from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from backend.decision.models import MLSignal, IntelligenceSnapshot
from backend.risk.models import RiskContext
from backend.positioning.models import PositionSizingContext
from backend.execution_authorization.models import ExecutionContext
from backend.execution_adapter.models import PaperExecutionContext


class TradingCycleStatus(Enum):
    COMPLETED = "COMPLETED"
    NO_PROPOSAL = "NO_PROPOSAL"
    FUSION_REJECTED = "FUSION_REJECTED"
    RISK_REJECTED = "RISK_REJECTED"
    SIZING_REJECTED = "SIZING_REJECTED"
    EXECUTION_AUTHORIZATION_REJECTED = "EXECUTION_AUTHORIZATION_REJECTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    PORTFOLIO_UPDATE_FAILED = "PORTFOLIO_UPDATE_FAILED"
    LIFECYCLE_REGISTRATION_FAILED = "LIFECYCLE_REGISTRATION_FAILED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TradingCycleInput:
    ml_signal: MLSignal
    risk_context: RiskContext
    position_sizing_context: PositionSizingContext
    execution_context: ExecutionContext
    paper_execution_context: PaperExecutionContext
    timestamp: str  # ISO 8601 format
    intelligence_snapshot: Optional[IntelligenceSnapshot] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradingCycleResult:
    cycle_id: str
    symbol: str
    timeframe: str
    status: TradingCycleStatus
    
    # Lineage IDs
    fusion_id: Optional[str] = None
    proposal_id: Optional[str] = None
    risk_authorization_id: Optional[str] = None
    sizing_id: Optional[str] = None
    execution_authorization_id: Optional[str] = None
    intent_id: Optional[str] = None
    execution_id: Optional[str] = None
    fill_ids: List[str] = field(default_factory=list)
    
    # Audit Flags
    intelligence_used: bool = False
    proposal_generated: bool = False
    risk_authorized: bool = False
    execution_authorized: bool = False
    executed: bool = False
    portfolio_updated: bool = False
    lifecycle_registered: bool = False
    
    # Failure Details
    rejection_stage: Optional[str] = None
    failed_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    # Latency & Timestamps
    started_at: str = ""
    completed_at: str = ""
    latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    policy_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
