from sqlalchemy import Table, Column, String, Integer, Float, Numeric, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

# Re-use project DeclarativeBase if available, otherwise define new Base
Base = declarative_base()


class SQLAlchemyRuntimeSession(Base):
    __tablename__ = "runtime_sessions"

    session_id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyTradingCycle(Base):
    __tablename__ = "trading_cycles"

    cycle_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("runtime_sessions.session_id"), nullable=False, index=True)
    cycle_index = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=False)
    latency_ms = Column(Float, nullable=False)
    total_latency_ms = Column(Float, nullable=False)
    rejection_stage = Column(String, nullable=True)
    failed_stage = Column(String, nullable=True)
    rejection_reason = Column(String, nullable=True)
    fusion_id = Column(String, nullable=True)
    proposal_id = Column(String, nullable=True)
    risk_authorization_id = Column(String, nullable=True)
    sizing_id = Column(String, nullable=True)
    execution_authorization_id = Column(String, nullable=True)
    intent_id = Column(String, nullable=True)
    execution_id = Column(String, nullable=True)
    fill_ids = Column(JSON, nullable=False, default=list)
    intelligence_used = Column(Boolean, nullable=False, default=False)
    proposal_generated = Column(Boolean, nullable=False, default=False)
    risk_authorized = Column(Boolean, nullable=False, default=False)
    execution_authorized = Column(Boolean, nullable=False, default=False)
    executed = Column(Boolean, nullable=False, default=False)
    portfolio_updated = Column(Boolean, nullable=False, default=False)
    lifecycle_registered = Column(Boolean, nullable=False, default=False)
    stage_timings = Column(JSON, nullable=False, default=dict)
    policy_version = Column(String, nullable=False, default="")
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyDecision(Base):
    __tablename__ = "decisions"

    decision_id = Column(String, primary_key=True)
    cycle_id = Column(String, ForeignKey("trading_cycles.cycle_id"), nullable=False, index=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    fusion_score = Column(Float, nullable=False)
    agreement_score = Column(Float, nullable=False)
    ml_contribution = Column(Float, nullable=False)
    intelligence_contribution = Column(Float, nullable=False)
    intelligence_used = Column(Boolean, nullable=False, default=False)
    intelligence_age_seconds = Column(Float, nullable=True)
    policy_version = Column(String, nullable=False)
    source_model_version = Column(String, nullable=False)
    reasoning_request_id = Column(String, nullable=True)
    risk_flags = Column(JSON, nullable=False, default=list)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyTradeProposal(Base):
    __tablename__ = "trade_proposals"

    proposal_id = Column(String, primary_key=True)
    decision_id = Column(String, nullable=True)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    fusion_score = Column(Float, nullable=False)
    source_model_version = Column(String, nullable=False)
    fusion_policy_version = Column(String, nullable=False)
    reasoning_request_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    risk_flags = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyRiskAuthorization(Base):
    __tablename__ = "risk_authorizations"

    authorization_id = Column(String, primary_key=True)
    proposal_id = Column(String, ForeignKey("trade_proposals.proposal_id"), nullable=False, index=True)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    status = Column(String, nullable=False)
    original_confidence = Column(Float, nullable=False)
    effective_confidence = Column(Float, nullable=False)
    rejection_reasons = Column(JSON, nullable=False, default=list)
    adjustment_reasons = Column(JSON, nullable=False, default=list)
    triggered_rules = Column(JSON, nullable=False, default=list)
    policy_version = Column(String, nullable=False)
    source_model_version = Column(String, nullable=False)
    fusion_policy_version = Column(String, nullable=False)
    proposal_created_at = Column(DateTime(timezone=True), nullable=False)
    evaluated_at = Column(DateTime(timezone=True), nullable=False)
    latency_ms = Column(Float, nullable=False)
    requested_risk_fraction = Column(Float, nullable=False)
    authorized_risk_fraction = Column(Float, nullable=False)
    reasoning_request_id = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyPositionSizing(Base):
    __tablename__ = "position_sizings"

    sizing_id = Column(String, primary_key=True)
    proposal_id = Column(String, ForeignKey("trade_proposals.proposal_id"), nullable=False, index=True)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    position_notional = Column(Numeric(precision=24, scale=8), nullable=False)
    entry_price = Column(Numeric(precision=24, scale=8), nullable=False)
    stop_loss_price = Column(Numeric(precision=24, scale=8), nullable=False)
    stop_distance_absolute = Column(Numeric(precision=24, scale=8), nullable=False)
    stop_distance_fraction = Column(Numeric(precision=24, scale=8), nullable=False)
    authorized_risk_fraction = Column(Numeric(precision=24, scale=8), nullable=False)
    risk_amount = Column(Numeric(precision=24, scale=8), nullable=False)
    leverage = Column(Numeric(precision=24, scale=8), nullable=False)
    estimated_margin_required = Column(Numeric(precision=24, scale=8), nullable=False)
    policy_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    authorization_id = Column(String, nullable=True)
    source_model_version = Column(String, nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyOrderIntent(Base):
    __tablename__ = "order_intents"

    intent_id = Column(String, primary_key=True)
    idempotency_key = Column(String, nullable=False)
    proposal_id = Column(String, ForeignKey("trade_proposals.proposal_id"), nullable=False, index=True)
    risk_authorization_id = Column(String, nullable=False)
    sizing_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    order_type = Column(String, nullable=False)
    limit_price = Column(Numeric(precision=24, scale=8), nullable=True)
    stop_loss = Column(Numeric(precision=24, scale=8), nullable=True)
    take_profit = Column(Numeric(precision=24, scale=8), nullable=True)
    environment = Column(String, nullable=False)
    source_model_version = Column(String, nullable=False)
    fusion_policy_version = Column(String, nullable=False)
    risk_policy_version = Column(String, nullable=False)
    position_sizing_policy_version = Column(String, nullable=False)
    execution_policy_version = Column(String, nullable=False)
    reasoning_request_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyExecution(Base):
    __tablename__ = "executions"

    execution_id = Column(String, primary_key=True)
    intent_id = Column(String, ForeignKey("order_intents.intent_id"), nullable=False, index=True)
    proposal_id = Column(String, nullable=False)
    risk_authorization_id = Column(String, nullable=False)
    sizing_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    requested_quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    filled_quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    average_fill_price = Column(Numeric(precision=24, scale=8), nullable=False)
    total_notional = Column(Numeric(precision=24, scale=8), nullable=False)
    total_fees = Column(Numeric(precision=24, scale=8), nullable=False)
    total_slippage = Column(Numeric(precision=24, scale=8), nullable=False)
    status = Column(String, nullable=False)
    rejection_reason = Column(String, nullable=False, default="")
    adapter_name = Column(String, nullable=False, default="")
    environment = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=False)
    latency_ms = Column(Float, nullable=False)
    policy_version = Column(String, nullable=False, default="")
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyFill(Base):
    __tablename__ = "fills"

    fill_id = Column(String, primary_key=True)
    execution_id = Column(String, ForeignKey("executions.execution_id"), nullable=False, index=True)
    intent_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    price = Column(Numeric(precision=24, scale=8), nullable=False)
    notional = Column(Numeric(precision=24, scale=8), nullable=False)
    fee = Column(Numeric(precision=24, scale=8), nullable=False)
    slippage_amount = Column(Numeric(precision=24, scale=8), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyPortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    portfolio_snapshot_id = Column(String, primary_key=True)
    portfolio_id = Column(String, nullable=False)
    initial_balance = Column(Numeric(precision=24, scale=8), nullable=False)
    cash_balance = Column(Numeric(precision=24, scale=8), nullable=False)
    equity = Column(Numeric(precision=24, scale=8), nullable=False)
    realized_pnl = Column(Numeric(precision=24, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=24, scale=8), nullable=False)
    total_fees = Column(Numeric(precision=24, scale=8), nullable=False)
    used_margin = Column(Numeric(precision=24, scale=8), nullable=False)
    available_balance = Column(Numeric(precision=24, scale=8), nullable=False)
    gross_exposure = Column(Numeric(precision=24, scale=8), nullable=False)
    net_exposure = Column(Numeric(precision=24, scale=8), nullable=False)
    open_position_count = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyPortfolioPosition(Base):
    __tablename__ = "portfolio_snapshot_positions"

    position_id = Column(String, primary_key=True)
    portfolio_snapshot_id = Column(String, ForeignKey("portfolio_snapshots.portfolio_snapshot_id"), nullable=False, index=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    average_entry_price = Column(Numeric(precision=24, scale=8), nullable=False)
    current_price = Column(Numeric(precision=24, scale=8), nullable=False)
    position_notional = Column(Numeric(precision=24, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=24, scale=8), nullable=False)
    realized_pnl = Column(Numeric(precision=24, scale=8), nullable=False)
    accumulated_fees = Column(Numeric(precision=24, scale=8), nullable=False)
    leverage = Column(Numeric(precision=24, scale=8), nullable=False)
    margin_used = Column(Numeric(precision=24, scale=8), nullable=False)
    opened_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    source_execution_ids = Column(JSON, nullable=False, default=list)
    source_fill_ids = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyPositionLifecycle(Base):
    __tablename__ = "position_lifecycle_states"

    lifecycle_id = Column(String, primary_key=True)
    position_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Numeric(precision=24, scale=8), nullable=False)
    average_entry_price = Column(Numeric(precision=24, scale=8), nullable=False)
    stop_loss = Column(Numeric(precision=24, scale=8), nullable=True)
    take_profit = Column(Numeric(precision=24, scale=8), nullable=True)
    trailing_stop_enabled = Column(Boolean, nullable=False)
    trailing_distance = Column(Numeric(precision=24, scale=8), nullable=True)
    trailing_activation_price = Column(Numeric(precision=24, scale=8), nullable=True)
    highest_price_since_entry = Column(Numeric(precision=24, scale=8), nullable=True)
    lowest_price_since_entry = Column(Numeric(precision=24, scale=8), nullable=True)
    active_trailing_stop_price = Column(Numeric(precision=24, scale=8), nullable=True)
    status = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    policy_version = Column(String, nullable=False)
    source_proposal_id = Column(String, nullable=True)
    source_execution_id = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)


class SQLAlchemyAuditEvent(Base):
    __tablename__ = "audit_events"

    audit_id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    cycle_id = Column(String, nullable=True, index=True)
    event_type = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    symbol = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    source_component = Column(String, nullable=False)
    status = Column(String, nullable=False)
    payload_json = Column(JSON, nullable=False)
    previous_hash = Column(String, nullable=False, default="")
    hash = Column(String, nullable=False, default="")
