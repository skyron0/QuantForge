import json
import hashlib
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from backend.persistence.exceptions import PersistenceValidationError


def _serialize_decimal(obj: Any) -> Any:
    """Helper to convert decimals to floats/strings for JSON serialization where needed."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_decimal(x) for x in obj]
    return obj


def validate_utc_timestamp(ts: str) -> str:
    """Enforces ISO 8601 UTC-aware timestamp validation, returns normalized ISO string."""
    if not ts:
        raise PersistenceValidationError("Timestamp cannot be empty")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise PersistenceValidationError(f"Timestamp must be UTC-aware, got naive: {ts}")
        # Enforce UTC timezone conversion/check
        return dt.astimezone(timezone.utc).isoformat()
    except Exception as e:
        raise PersistenceValidationError(f"Invalid timestamp format: {ts}. Error: {str(e)}")


def to_decimal(val: Any) -> Decimal:
    """Safely converts input value to Decimal, preserving precision."""
    if val is None:
        raise PersistenceValidationError("Monetary value cannot be None")
    try:
        d = Decimal(str(val))
        if d.is_nan() or d.is_infinite():
            raise PersistenceValidationError(f"Invalid decimal value: {val}")
        return d
    except Exception as e:
        raise PersistenceValidationError(f"Cannot convert {val} to Decimal. Error: {str(e)}")


def to_decimal_opt(val: Any) -> Optional[Decimal]:
    """Helper for optional decimal properties."""
    return to_decimal(val) if val is not None else None


@dataclass(frozen=True)
class RuntimeSessionRecord:
    session_id: str
    status: str
    started_at: str
    stopped_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.session_id:
            raise PersistenceValidationError("session_id is mandatory")
        object.__setattr__(self, "started_at", validate_utc_timestamp(self.started_at))
        if self.stopped_at is not None:
            object.__setattr__(self, "stopped_at", validate_utc_timestamp(self.stopped_at))


@dataclass(frozen=True)
class TradingCycleRecord:
    cycle_id: str
    session_id: str
    cycle_index: int
    status: str
    started_at: str
    completed_at: str
    latency_ms: float
    total_latency_ms: float
    rejection_stage: Optional[str] = None
    failed_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    fusion_id: Optional[str] = None
    proposal_id: Optional[str] = None
    risk_authorization_id: Optional[str] = None
    sizing_id: Optional[str] = None
    execution_authorization_id: Optional[str] = None
    intent_id: Optional[str] = None
    execution_id: Optional[str] = None
    fill_ids: List[str] = field(default_factory=list)
    intelligence_used: bool = False
    proposal_generated: bool = False
    risk_authorized: bool = False
    execution_authorized: bool = False
    executed: bool = False
    portfolio_updated: bool = False
    lifecycle_registered: bool = False
    stage_timings: Dict[str, float] = field(default_factory=dict)
    policy_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cycle_id or not self.session_id:
            raise PersistenceValidationError("cycle_id and session_id are mandatory")
        object.__setattr__(self, "started_at", validate_utc_timestamp(self.started_at))
        object.__setattr__(self, "completed_at", validate_utc_timestamp(self.completed_at))


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    cycle_id: str
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

    def __post_init__(self):
        if not self.decision_id or not self.cycle_id or not self.symbol:
            raise PersistenceValidationError("decision_id, cycle_id and symbol are mandatory")
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))


@dataclass(frozen=True)
class TradeProposalRecord:
    proposal_id: str
    decision_id: Optional[str]  # Optional link to FusionResult decision_id (same as fusion_id)
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

    def __post_init__(self):
        if not self.proposal_id or not self.symbol:
            raise PersistenceValidationError("proposal_id and symbol are mandatory")
        object.__setattr__(self, "created_at", validate_utc_timestamp(self.created_at))
        object.__setattr__(self, "expires_at", validate_utc_timestamp(self.expires_at))


@dataclass(frozen=True)
class RiskAuthorizationRecord:
    authorization_id: str
    proposal_id: str
    symbol: str
    direction: str
    status: str  # APPROVED, REJECTED, ADJUSTED
    original_confidence: float
    effective_confidence: float
    rejection_reasons: List[str]
    adjustment_reasons: List[str]
    triggered_rules: List[str]
    policy_version: str
    source_model_version: str
    fusion_policy_version: str
    proposal_created_at: str
    evaluated_at: str
    latency_ms: float
    requested_risk_fraction: float
    authorized_risk_fraction: float
    reasoning_request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.authorization_id or not self.proposal_id or not self.symbol:
            raise PersistenceValidationError("authorization_id, proposal_id and symbol are mandatory")
        object.__setattr__(self, "proposal_created_at", validate_utc_timestamp(self.proposal_created_at))
        object.__setattr__(self, "evaluated_at", validate_utc_timestamp(self.evaluated_at))


@dataclass(frozen=True)
class PositionSizingRecord:
    sizing_id: str
    proposal_id: str
    symbol: str
    direction: str
    quantity: Decimal
    position_notional: Decimal
    entry_price: Decimal
    stop_loss_price: Decimal
    stop_distance_absolute: Decimal
    stop_distance_fraction: Decimal
    authorized_risk_fraction: Decimal
    risk_amount: Decimal
    leverage: Decimal
    estimated_margin_required: Decimal
    policy_version: str
    created_at: str
    authorization_id: Optional[str]
    source_model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sizing_id or not self.proposal_id or not self.symbol:
            raise PersistenceValidationError("sizing_id, proposal_id and symbol are mandatory")
        # Enforce Decimal safety
        object.__setattr__(self, "quantity", to_decimal(self.quantity))
        object.__setattr__(self, "position_notional", to_decimal(self.position_notional))
        object.__setattr__(self, "entry_price", to_decimal(self.entry_price))
        object.__setattr__(self, "stop_loss_price", to_decimal(self.stop_loss_price))
        object.__setattr__(self, "stop_distance_absolute", to_decimal(self.stop_distance_absolute))
        object.__setattr__(self, "stop_distance_fraction", to_decimal(self.stop_distance_fraction))
        object.__setattr__(self, "authorized_risk_fraction", to_decimal(self.authorized_risk_fraction))
        object.__setattr__(self, "risk_amount", to_decimal(self.risk_amount))
        object.__setattr__(self, "leverage", to_decimal(self.leverage))
        object.__setattr__(self, "estimated_margin_required", to_decimal(self.estimated_margin_required))
        object.__setattr__(self, "created_at", validate_utc_timestamp(self.created_at))


@dataclass(frozen=True)
class OrderIntentRecord:
    intent_id: str
    idempotency_key: str
    proposal_id: str
    risk_authorization_id: str
    sizing_id: str
    symbol: str
    direction: str
    quantity: Decimal
    order_type: str
    limit_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    environment: str
    source_model_version: str
    fusion_policy_version: str
    risk_policy_version: str
    position_sizing_policy_version: str
    execution_policy_version: str
    reasoning_request_id: Optional[str]
    created_at: str
    expires_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.intent_id or not self.idempotency_key or not self.proposal_id:
            raise PersistenceValidationError("intent_id, idempotency_key and proposal_id are mandatory")
        object.__setattr__(self, "quantity", to_decimal(self.quantity))
        object.__setattr__(self, "limit_price", to_decimal_opt(self.limit_price))
        object.__setattr__(self, "stop_loss", to_decimal_opt(self.stop_loss))
        object.__setattr__(self, "take_profit", to_decimal_opt(self.take_profit))
        object.__setattr__(self, "created_at", validate_utc_timestamp(self.created_at))
        object.__setattr__(self, "expires_at", validate_utc_timestamp(self.expires_at))


@dataclass(frozen=True)
class FillRecord:
    fill_id: str
    execution_id: str
    intent_id: str
    symbol: str
    direction: str
    quantity: Decimal
    price: Decimal
    notional: Decimal
    fee: Decimal
    slippage_amount: Decimal
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.fill_id or not self.execution_id or not self.intent_id:
            raise PersistenceValidationError("fill_id, execution_id and intent_id are mandatory")
        object.__setattr__(self, "quantity", to_decimal(self.quantity))
        object.__setattr__(self, "price", to_decimal(self.price))
        object.__setattr__(self, "notional", to_decimal(self.notional))
        object.__setattr__(self, "fee", to_decimal(self.fee))
        object.__setattr__(self, "slippage_amount", to_decimal(self.slippage_amount))
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    intent_id: str
    proposal_id: str
    risk_authorization_id: str
    sizing_id: str
    symbol: str
    direction: str
    requested_quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal
    total_notional: Decimal
    total_fees: Decimal
    total_slippage: Decimal
    status: str
    rejection_reason: str
    adapter_name: str
    environment: str
    started_at: str
    completed_at: str
    latency_ms: float
    policy_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # The domain model includes nested fills, but we decouple writes. We can store them in database database/schema.py cleanly.

    def __post_init__(self):
        if not self.execution_id or not self.intent_id or not self.symbol:
            raise PersistenceValidationError("execution_id, intent_id and symbol are mandatory")
        object.__setattr__(self, "requested_quantity", to_decimal(self.requested_quantity))
        object.__setattr__(self, "filled_quantity", to_decimal(self.filled_quantity))
        object.__setattr__(self, "average_fill_price", to_decimal(self.average_fill_price))
        object.__setattr__(self, "total_notional", to_decimal(self.total_notional))
        object.__setattr__(self, "total_fees", to_decimal(self.total_fees))
        object.__setattr__(self, "total_slippage", to_decimal(self.total_slippage))
        object.__setattr__(self, "started_at", validate_utc_timestamp(self.started_at))
        object.__setattr__(self, "completed_at", validate_utc_timestamp(self.completed_at))


@dataclass(frozen=True)
class PositionSnapshotRecord:
    """Representing an entry in a portfolio's positions history."""
    position_id: str
    portfolio_snapshot_id: str
    symbol: str
    side: str
    quantity: Decimal
    average_entry_price: Decimal
    current_price: Decimal
    position_notional: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    accumulated_fees: Decimal
    leverage: Decimal
    margin_used: Decimal
    opened_at: str
    updated_at: str
    source_execution_ids: List[str] = field(default_factory=list)
    source_fill_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.position_id or not self.portfolio_snapshot_id or not self.symbol:
            raise PersistenceValidationError("position_id, portfolio_snapshot_id and symbol are mandatory")
        object.__setattr__(self, "quantity", to_decimal(self.quantity))
        object.__setattr__(self, "average_entry_price", to_decimal(self.average_entry_price))
        object.__setattr__(self, "current_price", to_decimal(self.current_price))
        object.__setattr__(self, "position_notional", to_decimal(self.position_notional))
        object.__setattr__(self, "unrealized_pnl", to_decimal(self.unrealized_pnl))
        object.__setattr__(self, "realized_pnl", to_decimal(self.realized_pnl))
        object.__setattr__(self, "accumulated_fees", to_decimal(self.accumulated_fees))
        object.__setattr__(self, "leverage", to_decimal(self.leverage))
        object.__setattr__(self, "margin_used", to_decimal(self.margin_used))
        object.__setattr__(self, "opened_at", validate_utc_timestamp(self.opened_at))
        object.__setattr__(self, "updated_at", validate_utc_timestamp(self.updated_at))


@dataclass(frozen=True)
class PortfolioSnapshotRecord:
    portfolio_snapshot_id: str
    portfolio_id: str
    initial_balance: Decimal
    cash_balance: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    used_margin: Decimal
    available_balance: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    open_position_count: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    positions: List[PositionSnapshotRecord] = field(default_factory=list)

    def __post_init__(self):
        if not self.portfolio_snapshot_id or not self.portfolio_id:
            raise PersistenceValidationError("portfolio_snapshot_id and portfolio_id are mandatory")
        object.__setattr__(self, "initial_balance", to_decimal(self.initial_balance))
        object.__setattr__(self, "cash_balance", to_decimal(self.cash_balance))
        object.__setattr__(self, "equity", to_decimal(self.equity))
        object.__setattr__(self, "realized_pnl", to_decimal(self.realized_pnl))
        object.__setattr__(self, "unrealized_pnl", to_decimal(self.unrealized_pnl))
        object.__setattr__(self, "total_fees", to_decimal(self.total_fees))
        object.__setattr__(self, "used_margin", to_decimal(self.used_margin))
        object.__setattr__(self, "available_balance", to_decimal(self.available_balance))
        object.__setattr__(self, "gross_exposure", to_decimal(self.gross_exposure))
        object.__setattr__(self, "net_exposure", to_decimal(self.net_exposure))
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))


@dataclass(frozen=True)
class PositionLifecycleRecord:
    lifecycle_id: str
    position_id: str
    symbol: str
    side: str
    quantity: Decimal
    average_entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    trailing_stop_enabled: bool
    trailing_distance: Optional[Decimal]
    trailing_activation_price: Optional[Decimal]
    highest_price_since_entry: Optional[Decimal]
    lowest_price_since_entry: Optional[Decimal]
    active_trailing_stop_price: Optional[Decimal]
    status: str
    created_at: str
    updated_at: str
    policy_version: str
    source_proposal_id: Optional[str] = None
    source_execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.lifecycle_id or not self.position_id or not self.symbol:
            raise PersistenceValidationError("lifecycle_id, position_id and symbol are mandatory")
        object.__setattr__(self, "quantity", to_decimal(self.quantity))
        object.__setattr__(self, "average_entry_price", to_decimal(self.average_entry_price))
        object.__setattr__(self, "stop_loss", to_decimal_opt(self.stop_loss))
        object.__setattr__(self, "take_profit", to_decimal_opt(self.take_profit))
        object.__setattr__(self, "trailing_distance", to_decimal_opt(self.trailing_distance))
        object.__setattr__(self, "trailing_activation_price", to_decimal_opt(self.trailing_activation_price))
        object.__setattr__(self, "highest_price_since_entry", to_decimal_opt(self.highest_price_since_entry))
        object.__setattr__(self, "lowest_price_since_entry", to_decimal_opt(self.lowest_price_since_entry))
        object.__setattr__(self, "active_trailing_stop_price", to_decimal_opt(self.active_trailing_stop_price))
        object.__setattr__(self, "created_at", validate_utc_timestamp(self.created_at))
        object.__setattr__(self, "updated_at", validate_utc_timestamp(self.updated_at))


@dataclass(frozen=True)
class AuditEventRecord:
    audit_id: str
    session_id: str
    cycle_id: Optional[str]
    event_type: str
    entity_type: str
    entity_id: str
    symbol: Optional[str]
    timestamp: str
    source_component: str
    status: str
    payload: Dict[str, Any]
    previous_hash: str = ""
    hash: str = ""

    def __post_init__(self):
        if not self.audit_id or not self.session_id or not self.event_type:
            raise PersistenceValidationError("audit_id, session_id and event_type are mandatory/required fields")
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))

    def compute_hash(self) -> str:
        """Determinstic serialization SHA-256 canonical hashing."""
        serialized_payload = json.dumps(_serialize_decimal(self.payload), sort_keys=True)
        # Combine canonical elements
        content_string = (
            f"audit_id:{self.audit_id}|"
            f"session_id:{self.session_id}|"
            f"cycle_id:{self.cycle_id or ''}|"
            f"event_type:{self.event_type}|"
            f"entity_type:{self.entity_type}|"
            f"entity_id:{self.entity_id}|"
            f"symbol:{self.symbol or ''}|"
            f"timestamp:{self.timestamp}|"
            f"source_component:{self.source_component}|"
            f"status:{self.status}|"
            f"payload:{serialized_payload}|"
            f"previous_hash:{self.previous_hash}"
        )
        return hashlib.sha256(content_string.encode('utf-8')).hexdigest()
