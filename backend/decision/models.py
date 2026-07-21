from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class MLSignal:
    model_version: str
    symbol: str
    timeframe: str
    prediction: float
    direction: str
    confidence: float
    calibrated: bool
    timestamp: str
    drift_status: str  # e.g., "normal", "warning", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntelligenceSnapshot:
    snapshot_id: str
    symbol: str
    timeframe: str
    market_regime: str
    directional_bias: str
    confidence: float
    risk_flags: List[str]
    evidence: List[str]
    reasoning_summary: str
    provider: str
    model: str
    request_id: str
    prompt_id: str
    prompt_version: str
    generated_at: str
    expires_at: str
    latency_ms: float


@dataclass(frozen=True)
class FusionInput:
    ml_signal: MLSignal
    timestamp: str
    intelligence_snapshot: Optional[IntelligenceSnapshot] = None
    market_context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionResult:
    fusion_id: str
    symbol: str
    timeframe: str
    direction: str
    confidence: float
    fusion_score: float
    agreement_score: float
    ml_contribution: float
    intelligence_contribution: float
    intelligence_used: bool
    intelligence_age_seconds: Optional[float]
    policy_version: str
    source_model_version: str
    reasoning_request_id: Optional[str]
    risk_flags: List[str]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeProposal:
    proposal_id: str
    symbol: str
    direction: str
    confidence: float
    fusion_score: float
    source_model_version: str
    fusion_policy_version: str
    reasoning_request_id: Optional[str]
    created_at: str
    expires_at: str
    risk_flags: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    action: str
    confidence: float
    reason: str