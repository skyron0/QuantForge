from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class AIRequest:
    request_id: str
    task_type: str
    system_prompt: str
    user_prompt: str
    context: Optional[Dict[str, Any]] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    response_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    request_id: str
    provider: str
    model: str
    content: str
    structured_output: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningRequest:
    symbol: str
    timeframe: str
    timestamp: str  # ISO-8601 string
    features: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    market_context: Dict[str, Any]
    portfolio_context: Dict[str, Any]
    risk_context: Dict[str, Any]
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    market_regime: str
    directional_bias: str
    confidence: float
    risk_flags: List[str]
    reasoning_summary: str
    evidence: List[str]
    provider: str
    model: str
    latency_ms: float
    request_id: str
    prompt_id: str
    prompt_version: str
    timestamp: str  # ISO-8601 string


@dataclass
class ProviderHealth:
    available: bool
    provider: str
    model: str
    latency_ms: float
    error: Optional[str] = None
