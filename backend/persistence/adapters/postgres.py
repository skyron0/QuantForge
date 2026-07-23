from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional
from backend.persistence.exceptions import DuplicateRecordError, RecordNotFoundError, PersistenceWriteError, PersistenceReadError
from backend.persistence.repositories import (
    RuntimeSessionRepository, TradingCycleRepository, DecisionRepository,
    TradeProposalRepository, RiskAuthorizationRepository, PositionSizingRepository,
    OrderIntentRepository, ExecutionRepository, PortfolioRepository,
    PositionLifecycleRepository, AuditRepository
)
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord, DecisionRecord,
    TradeProposalRecord, RiskAuthorizationRecord, PositionSizingRecord,
    OrderIntentRecord, ExecutionRecord, FillRecord,
    PortfolioSnapshotRecord, PositionSnapshotRecord, PositionLifecycleRecord, AuditEventRecord
)
from backend.persistence.database.schema import (
    SQLAlchemyRuntimeSession, SQLAlchemyTradingCycle, SQLAlchemyDecision,
    SQLAlchemyTradeProposal, SQLAlchemyRiskAuthorization, SQLAlchemyPositionSizing,
    SQLAlchemyOrderIntent, SQLAlchemyExecution, SQLAlchemyFill,
    SQLAlchemyPortfolioSnapshot, SQLAlchemyPortfolioPosition, SQLAlchemyPositionLifecycle,
    SQLAlchemyAuditEvent
)


class PostgresRuntimeSessionRepository(RuntimeSessionRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: RuntimeSessionRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyRuntimeSession).filter_by(session_id=record.session_id).first()
            if existing:
                # Compare fields
                if (existing.status != record.status or 
                    existing.metadata_json != record.metadata):
                    raise DuplicateRecordError(f"Conflict: Session {record.session_id} exists with different data.")
                return
            
            db_obj = SQLAlchemyRuntimeSession(
                session_id=record.session_id,
                status=record.status,
                started_at=record.started_at,
                stopped_at=record.stopped_at,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save Session: {str(e)}")

    def get_by_id(self, session_id: str) -> Optional[RuntimeSessionRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyRuntimeSession).filter_by(session_id=session_id).first()
            if not db_obj:
                return None
            return RuntimeSessionRecord(
                session_id=db_obj.session_id,
                status=db_obj.status,
                started_at=db_obj.started_at.isoformat(),
                stopped_at=db_obj.stopped_at.isoformat() if db_obj.stopped_at else None,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read Session: {str(e)}")


class PostgresTradingCycleRepository(TradingCycleRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: TradingCycleRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyTradingCycle).filter_by(cycle_id=record.cycle_id).first()
            if existing:
                if (existing.session_id != record.session_id or
                    existing.cycle_index != record.cycle_index or
                    existing.status != record.status):
                    raise DuplicateRecordError(f"Conflict: TradingCycle {record.cycle_id} exists with different data.")
                return

            db_obj = SQLAlchemyTradingCycle(
                cycle_id=record.cycle_id,
                session_id=record.session_id,
                cycle_index=record.cycle_index,
                status=record.status,
                started_at=record.started_at,
                completed_at=record.completed_at,
                latency_ms=record.latency_ms,
                total_latency_ms=record.total_latency_ms,
                rejection_stage=record.rejection_stage,
                failed_stage=record.failed_stage,
                rejection_reason=record.rejection_reason,
                fusion_id=record.fusion_id,
                proposal_id=record.proposal_id,
                risk_authorization_id=record.risk_authorization_id,
                sizing_id=record.sizing_id,
                execution_authorization_id=record.execution_authorization_id,
                intent_id=record.intent_id,
                execution_id=record.execution_id,
                fill_ids=record.fill_ids,
                intelligence_used=record.intelligence_used,
                proposal_generated=record.proposal_generated,
                risk_authorized=record.risk_authorized,
                execution_authorized=record.execution_authorized,
                executed=record.executed,
                portfolio_updated=record.portfolio_updated,
                lifecycle_registered=record.lifecycle_registered,
                stage_timings=record.stage_timings,
                policy_version=record.policy_version,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save TradingCycle: {str(e)}")

    def get_by_id(self, cycle_id: str) -> Optional[TradingCycleRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyTradingCycle).filter_by(cycle_id=cycle_id).first()
            if not db_obj:
                return None
            return TradingCycleRecord(
                cycle_id=db_obj.cycle_id,
                session_id=db_obj.session_id,
                cycle_index=db_obj.cycle_index,
                status=db_obj.status,
                started_at=db_obj.started_at.isoformat(),
                completed_at=db_obj.completed_at.isoformat(),
                latency_ms=db_obj.latency_ms,
                total_latency_ms=db_obj.total_latency_ms,
                rejection_stage=db_obj.rejection_stage,
                failed_stage=db_obj.failed_stage,
                rejection_reason=db_obj.rejection_reason,
                fusion_id=db_obj.fusion_id,
                proposal_id=db_obj.proposal_id,
                risk_authorization_id=db_obj.risk_authorization_id,
                sizing_id=db_obj.sizing_id,
                execution_authorization_id=db_obj.execution_authorization_id,
                intent_id=db_obj.intent_id,
                execution_id=db_obj.execution_id,
                fill_ids=db_obj.fill_ids,
                intelligence_used=db_obj.intelligence_used,
                proposal_generated=db_obj.proposal_generated,
                risk_authorized=db_obj.risk_authorized,
                execution_authorized=db_obj.execution_authorized,
                executed=db_obj.executed,
                portfolio_updated=db_obj.portfolio_updated,
                lifecycle_registered=db_obj.lifecycle_registered,
                stage_timings=db_obj.stage_timings,
                policy_version=db_obj.policy_version,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read TradingCycle: {str(e)}")

    def list_by_session(self, session_id: str) -> List[TradingCycleRecord]:
        try:
            results = self.session.query(SQLAlchemyTradingCycle).filter_by(session_id=session_id).all()
            return [self.get_by_id(r.cycle_id) for r in results]
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to list TradingCycles: {str(e)}")


class PostgresDecisionRepository(DecisionRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: DecisionRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyDecision).filter_by(decision_id=record.decision_id).first()
            if existing:
                if (existing.cycle_id != record.cycle_id or
                    existing.symbol != record.symbol or
                    existing.direction != record.direction):
                    raise DuplicateRecordError(f"Conflict: Decision {record.decision_id} exists with different data.")
                return

            db_obj = SQLAlchemyDecision(
                decision_id=record.decision_id,
                cycle_id=record.cycle_id,
                symbol=record.symbol,
                timeframe=record.timeframe,
                direction=record.direction,
                confidence=record.confidence,
                fusion_score=record.fusion_score,
                agreement_score=record.agreement_score,
                ml_contribution=record.ml_contribution,
                intelligence_contribution=record.intelligence_contribution,
                intelligence_used=record.intelligence_used,
                intelligence_age_seconds=record.intelligence_age_seconds,
                policy_version=record.policy_version,
                source_model_version=record.source_model_version,
                reasoning_request_id=record.reasoning_request_id,
                risk_flags=record.risk_flags,
                timestamp=record.timestamp,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save Decision: {str(e)}")

    def get_by_id(self, decision_id: str) -> Optional[DecisionRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyDecision).filter_by(decision_id=decision_id).first()
            if not db_obj:
                return None
            return DecisionRecord(
                decision_id=db_obj.decision_id,
                cycle_id=db_obj.cycle_id,
                symbol=db_obj.symbol,
                timeframe=db_obj.timeframe,
                direction=db_obj.direction,
                confidence=db_obj.confidence,
                fusion_score=db_obj.fusion_score,
                agreement_score=db_obj.agreement_score,
                ml_contribution=db_obj.ml_contribution,
                intelligence_contribution=db_obj.intelligence_contribution,
                intelligence_used=db_obj.intelligence_used,
                intelligence_age_seconds=db_obj.intelligence_age_seconds,
                policy_version=db_obj.policy_version,
                source_model_version=db_obj.source_model_version,
                reasoning_request_id=db_obj.reasoning_request_id,
                risk_flags=db_obj.risk_flags,
                timestamp=db_obj.timestamp.isoformat(),
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read Decision: {str(e)}")

    def list_by_cycle(self, cycle_id: str) -> List[DecisionRecord]:
        try:
            results = self.session.query(SQLAlchemyDecision).filter_by(cycle_id=cycle_id).all()
            return [self.get_by_id(r.decision_id) for r in results]
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to list Decisions: {str(e)}")


class PostgresTradeProposalRepository(TradeProposalRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: TradeProposalRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyTradeProposal).filter_by(proposal_id=record.proposal_id).first()
            if existing:
                if (existing.symbol != record.symbol or
                    existing.direction != record.direction or
                    existing.confidence != record.confidence):
                    raise DuplicateRecordError(f"Conflict: TradeProposal {record.proposal_id} exists with different data.")
                return

            db_obj = SQLAlchemyTradeProposal(
                proposal_id=record.proposal_id,
                decision_id=record.decision_id,
                symbol=record.symbol,
                direction=record.direction,
                confidence=record.confidence,
                fusion_score=record.fusion_score,
                source_model_version=record.source_model_version,
                fusion_policy_version=record.fusion_policy_version,
                reasoning_request_id=record.reasoning_request_id,
                created_at=record.created_at,
                expires_at=record.expires_at,
                risk_flags=record.risk_flags,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save TradeProposal: {str(e)}")

    def get_by_id(self, proposal_id: str) -> Optional[TradeProposalRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyTradeProposal).filter_by(proposal_id=proposal_id).first()
            if not db_obj:
                return None
            return TradeProposalRecord(
                proposal_id=db_obj.proposal_id,
                decision_id=db_obj.decision_id,
                symbol=db_obj.symbol,
                direction=db_obj.direction,
                confidence=db_obj.confidence,
                fusion_score=db_obj.fusion_score,
                source_model_version=db_obj.source_model_version,
                fusion_policy_version=db_obj.fusion_policy_version,
                reasoning_request_id=db_obj.reasoning_request_id,
                created_at=db_obj.created_at.isoformat(),
                expires_at=db_obj.expires_at.isoformat(),
                risk_flags=db_obj.risk_flags,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read TradeProposal: {str(e)}")


class PostgresRiskAuthorizationRepository(RiskAuthorizationRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: RiskAuthorizationRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyRiskAuthorization).filter_by(authorization_id=record.authorization_id).first()
            if existing:
                if (existing.proposal_id != record.proposal_id or
                    existing.symbol != record.symbol or
                    existing.status != record.status):
                    raise DuplicateRecordError(f"Conflict: RiskAuthorization {record.authorization_id} exists.")
                return

            db_obj = SQLAlchemyRiskAuthorization(
                authorization_id=record.authorization_id,
                proposal_id=record.proposal_id,
                symbol=record.symbol,
                direction=record.direction,
                status=record.status,
                original_confidence=record.original_confidence,
                effective_confidence=record.effective_confidence,
                rejection_reasons=record.rejection_reasons,
                adjustment_reasons=record.adjustment_reasons,
                triggered_rules=record.triggered_rules,
                policy_version=record.policy_version,
                source_model_version=record.source_model_version,
                fusion_policy_version=record.fusion_policy_version,
                proposal_created_at=record.proposal_created_at,
                evaluated_at=record.evaluated_at,
                latency_ms=record.latency_ms,
                requested_risk_fraction=record.requested_risk_fraction,
                authorized_risk_fraction=record.authorized_risk_fraction,
                reasoning_request_id=record.reasoning_request_id,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save RiskAuthorization: {str(e)}")

    def get_by_id(self, authorization_id: str) -> Optional[RiskAuthorizationRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyRiskAuthorization).filter_by(authorization_id=authorization_id).first()
            if not db_obj:
                return None
            return RiskAuthorizationRecord(
                authorization_id=db_obj.authorization_id,
                proposal_id=db_obj.proposal_id,
                symbol=db_obj.symbol,
                direction=db_obj.direction,
                status=db_obj.status,
                original_confidence=db_obj.original_confidence,
                effective_confidence=db_obj.effective_confidence,
                rejection_reasons=db_obj.rejection_reasons,
                adjustment_reasons=db_obj.adjustment_reasons,
                triggered_rules=db_obj.triggered_rules,
                policy_version=db_obj.policy_version,
                source_model_version=db_obj.source_model_version,
                fusion_policy_version=db_obj.fusion_policy_version,
                proposal_created_at=db_obj.proposal_created_at.isoformat(),
                evaluated_at=db_obj.evaluated_at.isoformat(),
                latency_ms=db_obj.latency_ms,
                requested_risk_fraction=db_obj.requested_risk_fraction,
                authorized_risk_fraction=db_obj.authorized_risk_fraction,
                reasoning_request_id=db_obj.reasoning_request_id,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read RiskAuthorization: {str(e)}")


class PostgresPositionSizingRepository(PositionSizingRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: PositionSizingRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyPositionSizing).filter_by(sizing_id=record.sizing_id).first()
            if existing:
                if (existing.proposal_id != record.proposal_id or
                    existing.symbol != record.symbol or
                    existing.quantity != record.quantity):
                    raise DuplicateRecordError(f"Conflict: PositionSizing {record.sizing_id} exists.")
                return

            db_obj = SQLAlchemyPositionSizing(
                sizing_id=record.sizing_id,
                proposal_id=record.proposal_id,
                symbol=record.symbol,
                direction=record.direction,
                quantity=record.quantity,
                position_notional=record.position_notional,
                entry_price=record.entry_price,
                stop_loss_price=record.stop_loss_price,
                stop_distance_absolute=record.stop_distance_absolute,
                stop_distance_fraction=record.stop_distance_fraction,
                authorized_risk_fraction=record.authorized_risk_fraction,
                risk_amount=record.risk_amount,
                leverage=record.leverage,
                estimated_margin_required=record.estimated_margin_required,
                policy_version=record.policy_version,
                created_at=record.created_at,
                authorization_id=record.authorization_id,
                source_model_version=record.source_model_version,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save PositionSizing: {str(e)}")

    def get_by_id(self, sizing_id: str) -> Optional[PositionSizingRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyPositionSizing).filter_by(sizing_id=sizing_id).first()
            if not db_obj:
                return None
            return PositionSizingRecord(
                sizing_id=db_obj.sizing_id,
                proposal_id=db_obj.proposal_id,
                symbol=db_obj.symbol,
                direction=db_obj.direction,
                quantity=db_obj.quantity,
                position_notional=db_obj.position_notional,
                entry_price=db_obj.entry_price,
                stop_loss_price=db_obj.stop_loss_price,
                stop_distance_absolute=db_obj.stop_distance_absolute,
                stop_distance_fraction=db_obj.stop_distance_fraction,
                authorized_risk_fraction=db_obj.authorized_risk_fraction,
                risk_amount=db_obj.risk_amount,
                leverage=db_obj.leverage,
                estimated_margin_required=db_obj.estimated_margin_required,
                policy_version=db_obj.policy_version,
                created_at=db_obj.created_at.isoformat(),
                authorization_id=db_obj.authorization_id,
                source_model_version=db_obj.source_model_version,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read PositionSizing: {str(e)}")


class PostgresOrderIntentRepository(OrderIntentRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: OrderIntentRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyOrderIntent).filter_by(intent_id=record.intent_id).first()
            if existing:
                if (existing.idempotency_key != record.idempotency_key or
                    existing.proposal_id != record.proposal_id or
                    existing.symbol != record.symbol or
                    existing.quantity != record.quantity):
                    raise DuplicateRecordError(f"Conflict: OrderIntent {record.intent_id} exists.")
                return

            db_obj = SQLAlchemyOrderIntent(
                intent_id=record.intent_id,
                idempotency_key=record.idempotency_key,
                proposal_id=record.proposal_id,
                risk_authorization_id=record.risk_authorization_id,
                sizing_id=record.sizing_id,
                symbol=record.symbol,
                direction=record.direction,
                quantity=record.quantity,
                order_type=record.order_type,
                limit_price=record.limit_price,
                stop_loss=record.stop_loss,
                take_profit=record.take_profit,
                environment=record.environment,
                source_model_version=record.source_model_version,
                fusion_policy_version=record.fusion_policy_version,
                risk_policy_version=record.risk_policy_version,
                position_sizing_policy_version=record.position_sizing_policy_version,
                execution_policy_version=record.execution_policy_version,
                reasoning_request_id=record.reasoning_request_id,
                created_at=record.created_at,
                expires_at=record.expires_at,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save OrderIntent: {str(e)}")

    def get_by_id(self, intent_id: str) -> Optional[OrderIntentRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyOrderIntent).filter_by(intent_id=intent_id).first()
            if not db_obj:
                return None
            return OrderIntentRecord(
                intent_id=db_obj.intent_id,
                idempotency_key=db_obj.idempotency_key,
                proposal_id=db_obj.proposal_id,
                risk_authorization_id=db_obj.risk_authorization_id,
                sizing_id=db_obj.sizing_id,
                symbol=db_obj.symbol,
                direction=db_obj.direction,
                quantity=db_obj.quantity,
                order_type=db_obj.order_type,
                limit_price=db_obj.limit_price,
                stop_loss=db_obj.stop_loss,
                take_profit=db_obj.take_profit,
                environment=db_obj.environment,
                source_model_version=db_obj.source_model_version,
                fusion_policy_version=db_obj.fusion_policy_version,
                risk_policy_version=db_obj.risk_policy_version,
                position_sizing_policy_version=db_obj.position_sizing_policy_version,
                execution_policy_version=db_obj.execution_policy_version,
                reasoning_request_id=db_obj.reasoning_request_id,
                created_at=db_obj.created_at.isoformat(),
                expires_at=db_obj.expires_at.isoformat(),
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read OrderIntent: {str(e)}")


class PostgresExecutionRepository(ExecutionRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: ExecutionRecord, fills: List[FillRecord]) -> None:
        try:
            existing = self.session.query(SQLAlchemyExecution).filter_by(execution_id=record.execution_id).first()
            if existing:
                # Compare record properties
                if (existing.intent_id != record.intent_id or
                    existing.filled_quantity != record.filled_quantity or
                    existing.average_fill_price != record.average_fill_price):
                    raise DuplicateRecordError(f"Conflict: Execution {record.execution_id} exists with different data.")
                # Idempotent write checked successfully
                return

            db_obj = SQLAlchemyExecution(
                execution_id=record.execution_id,
                intent_id=record.intent_id,
                proposal_id=record.proposal_id,
                risk_authorization_id=record.risk_authorization_id,
                sizing_id=record.sizing_id,
                symbol=record.symbol,
                direction=record.direction,
                requested_quantity=record.requested_quantity,
                filled_quantity=record.filled_quantity,
                average_fill_price=record.average_fill_price,
                total_notional=record.total_notional,
                total_fees=record.total_fees,
                total_slippage=record.total_slippage,
                status=record.status,
                rejection_reason=record.rejection_reason,
                adapter_name=record.adapter_name,
                environment=record.environment,
                started_at=record.started_at,
                completed_at=record.completed_at,
                latency_ms=record.latency_ms,
                policy_version=record.policy_version,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)

            # Save nested fills
            for f in fills:
                fill_db = SQLAlchemyFill(
                    fill_id=f.fill_id,
                    execution_id=f.execution_id,
                    intent_id=f.intent_id,
                    symbol=f.symbol,
                    direction=f.direction,
                    quantity=f.quantity,
                    price=f.price,
                    notional=f.notional,
                    fee=f.fee,
                    slippage_amount=f.slippage_amount,
                    timestamp=f.timestamp,
                    metadata_json=f.metadata
                )
                self.session.add(fill_db)

            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save Execution & Fills: {str(e)}")

    def get_by_id(self, execution_id: str) -> Optional[ExecutionRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyExecution).filter_by(execution_id=execution_id).first()
            if not db_obj:
                return None
            return ExecutionRecord(
                execution_id=db_obj.execution_id,
                intent_id=db_obj.intent_id,
                proposal_id=db_obj.proposal_id,
                risk_authorization_id=db_obj.risk_authorization_id,
                sizing_id=db_obj.sizing_id,
                symbol=db_obj.symbol,
                direction=db_obj.direction,
                requested_quantity=db_obj.requested_quantity,
                filled_quantity=db_obj.filled_quantity,
                average_fill_price=db_obj.average_fill_price,
                total_notional=db_obj.total_notional,
                total_fees=db_obj.total_fees,
                total_slippage=db_obj.total_slippage,
                status=db_obj.status,
                rejection_reason=db_obj.rejection_reason,
                adapter_name=db_obj.adapter_name,
                environment=db_obj.environment,
                started_at=db_obj.started_at.isoformat(),
                completed_at=db_obj.completed_at.isoformat(),
                latency_ms=db_obj.latency_ms,
                policy_version=db_obj.policy_version,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read Execution: {str(e)}")

    def get_fills(self, execution_id: str) -> List[FillRecord]:
        try:
            db_fills = self.session.query(SQLAlchemyFill).filter_by(execution_id=execution_id).all()
            return [
                FillRecord(
                    fill_id=f.fill_id,
                    execution_id=f.execution_id,
                    intent_id=f.intent_id,
                    symbol=f.symbol,
                    direction=f.direction,
                    quantity=f.quantity,
                    price=f.price,
                    notional=f.notional,
                    fee=f.fee,
                    slippage_amount=f.slippage_amount,
                    timestamp=f.timestamp.isoformat(),
                    metadata=f.metadata_json
                )
                for f in db_fills
            ]
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read Fills: {str(e)}")


class PostgresPortfolioRepository(PortfolioRepository):
    def __init__(self, session: Session):
        self.session = session

    def save_snapshot(self, snapshot: PortfolioSnapshotRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyPortfolioSnapshot).filter_by(portfolio_snapshot_id=snapshot.portfolio_snapshot_id).first()
            if existing:
                if (existing.portfolio_id != snapshot.portfolio_id or
                    existing.equity != snapshot.equity):
                    raise DuplicateRecordError(f"Conflict: PortfolioSnapshot {snapshot.portfolio_snapshot_id} exists.")
                return

            db_obj = SQLAlchemyPortfolioSnapshot(
                portfolio_snapshot_id=snapshot.portfolio_snapshot_id,
                portfolio_id=snapshot.portfolio_id,
                initial_balance=snapshot.initial_balance,
                cash_balance=snapshot.cash_balance,
                equity=snapshot.equity,
                realized_pnl=snapshot.realized_pnl,
                unrealized_pnl=snapshot.unrealized_pnl,
                total_fees=snapshot.total_fees,
                used_margin=snapshot.used_margin,
                available_balance=snapshot.available_balance,
                gross_exposure=snapshot.gross_exposure,
                net_exposure=snapshot.net_exposure,
                open_position_count=snapshot.open_position_count,
                timestamp=snapshot.timestamp,
                metadata_json=snapshot.metadata
            )
            self.session.add(db_obj)

            # Save positions
            for pos in snapshot.positions:
                pos_db = SQLAlchemyPortfolioPosition(
                    position_id=pos.position_id,
                    portfolio_snapshot_id=pos.portfolio_snapshot_id,
                    symbol=pos.symbol,
                    side=pos.side,
                    quantity=pos.quantity,
                    average_entry_price=pos.average_entry_price,
                    current_price=pos.current_price,
                    position_notional=pos.position_notional,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    accumulated_fees=pos.accumulated_fees,
                    leverage=pos.leverage,
                    margin_used=pos.margin_used,
                    opened_at=pos.opened_at,
                    updated_at=pos.updated_at,
                    source_execution_ids=pos.source_execution_ids,
                    source_fill_ids=pos.source_fill_ids,
                    metadata_json=pos.metadata
                )
                self.session.add(pos_db)

            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save PortfolioSnapshot: {str(e)}")

    def get_snapshot(self, portfolio_snapshot_id: str) -> Optional[PortfolioSnapshotRecord]:
        try:
            db_snap = self.session.query(SQLAlchemyPortfolioSnapshot).filter_by(portfolio_snapshot_id=portfolio_snapshot_id).first()
            if not db_snap:
                return None

            db_positions = self.session.query(SQLAlchemyPortfolioPosition).filter_by(portfolio_snapshot_id=portfolio_snapshot_id).all()
            positions_rec = [
                PositionSnapshotRecord(
                    position_id=p.position_id,
                    portfolio_snapshot_id=p.portfolio_snapshot_id,
                    symbol=p.symbol,
                    side=p.side,
                    quantity=p.quantity,
                    average_entry_price=p.average_entry_price,
                    current_price=p.current_price,
                    position_notional=p.position_notional,
                    unrealized_pnl=p.unrealized_pnl,
                    realized_pnl=p.realized_pnl,
                    accumulated_fees=p.accumulated_fees,
                    leverage=p.leverage,
                    margin_used=p.margin_used,
                    opened_at=p.opened_at.isoformat(),
                    updated_at=p.updated_at.isoformat(),
                    source_execution_ids=p.source_execution_ids,
                    source_fill_ids=p.source_fill_ids,
                    metadata=p.metadata_json
                )
                for p in db_positions
            ]

            return PortfolioSnapshotRecord(
                portfolio_snapshot_id=db_snap.portfolio_snapshot_id,
                portfolio_id=db_snap.portfolio_id,
                initial_balance=db_snap.initial_balance,
                cash_balance=db_snap.cash_balance,
                equity=db_snap.equity,
                realized_pnl=db_snap.realized_pnl,
                unrealized_pnl=db_snap.unrealized_pnl,
                total_fees=db_snap.total_fees,
                used_margin=db_snap.used_margin,
                available_balance=db_snap.available_balance,
                gross_exposure=db_snap.gross_exposure,
                net_exposure=db_snap.net_exposure,
                open_position_count=db_snap.open_position_count,
                timestamp=db_snap.timestamp.isoformat(),
                metadata=db_snap.metadata_json,
                positions=positions_rec
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read PortfolioSnapshot: {str(e)}")

    def list_snapshots_by_portfolio(self, portfolio_id: str) -> List[PortfolioSnapshotRecord]:
        try:
            results = self.session.query(SQLAlchemyPortfolioSnapshot).filter_by(portfolio_id=portfolio_id).all()
            return [self.get_snapshot(r.portfolio_snapshot_id) for r in results]
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to list PortfolioSnapshots: {str(e)}")


class PostgresPositionLifecycleRepository(PositionLifecycleRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: PositionLifecycleRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyPositionLifecycle).filter_by(lifecycle_id=record.lifecycle_id).first()
            if existing:
                if (existing.position_id != record.position_id or
                    existing.symbol != record.symbol or
                    existing.status != record.status):
                    raise DuplicateRecordError(f"Conflict: PositionLifecycle {record.lifecycle_id} exists.")
                return

            db_obj = SQLAlchemyPositionLifecycle(
                lifecycle_id=record.lifecycle_id,
                position_id=record.position_id,
                symbol=record.symbol,
                side=record.side,
                quantity=record.quantity,
                average_entry_price=record.average_entry_price,
                stop_loss=record.stop_loss,
                take_profit=record.take_profit,
                trailing_stop_enabled=record.trailing_stop_enabled,
                trailing_distance=record.trailing_distance,
                trailing_activation_price=record.trailing_activation_price,
                highest_price_since_entry=record.highest_price_since_entry,
                lowest_price_since_entry=record.lowest_price_since_entry,
                active_trailing_stop_price=record.active_trailing_stop_price,
                status=record.status,
                created_at=record.created_at,
                updated_at=record.updated_at,
                policy_version=record.policy_version,
                source_proposal_id=record.source_proposal_id,
                source_execution_id=record.source_execution_id,
                metadata_json=record.metadata
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save PositionLifecycle: {str(e)}")

    def get_by_id(self, lifecycle_id: str) -> Optional[PositionLifecycleRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyPositionLifecycle).filter_by(lifecycle_id=lifecycle_id).first()
            if not db_obj:
                return None
            return PositionLifecycleRecord(
                lifecycle_id=db_obj.lifecycle_id,
                position_id=db_obj.position_id,
                symbol=db_obj.symbol,
                side=db_obj.side,
                quantity=db_obj.quantity,
                average_entry_price=db_obj.average_entry_price,
                stop_loss=db_obj.stop_loss,
                take_profit=db_obj.take_profit,
                trailing_stop_enabled=db_obj.trailing_stop_enabled,
                trailing_distance=db_obj.trailing_distance,
                trailing_activation_price=db_obj.trailing_activation_price,
                highest_price_since_entry=db_obj.highest_price_since_entry,
                lowest_price_since_entry=db_obj.lowest_price_since_entry,
                active_trailing_stop_price=db_obj.active_trailing_stop_price,
                status=db_obj.status,
                created_at=db_obj.created_at.isoformat(),
                updated_at=db_obj.updated_at.isoformat(),
                policy_version=db_obj.policy_version,
                source_proposal_id=db_obj.source_proposal_id,
                source_execution_id=db_obj.source_execution_id,
                metadata=db_obj.metadata_json
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read PositionLifecycle: {str(e)}")

    def list_by_symbol(self, symbol: str) -> List[PositionLifecycleRecord]:
        try:
            results = self.session.query(SQLAlchemyPositionLifecycle).filter_by(symbol=symbol).all()
            return [self.get_by_id(r.lifecycle_id) for r in results]
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to list PositionLifecycles: {str(e)}")


class PostgresAuditRepository(AuditRepository):
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: AuditEventRecord) -> None:
        try:
            existing = self.session.query(SQLAlchemyAuditEvent).filter_by(audit_id=record.audit_id).first()
            if existing:
                if (existing.session_id != record.session_id or
                    existing.hash != record.hash):
                    raise DuplicateRecordError(f"Conflict: AuditEvent {record.audit_id} already exists.")
                return

            db_obj = SQLAlchemyAuditEvent(
                audit_id=record.audit_id,
                session_id=record.session_id,
                cycle_id=record.cycle_id,
                event_type=record.event_type,
                entity_type=record.entity_type,
                entity_id=record.entity_id,
                symbol=record.symbol,
                timestamp=record.timestamp,
                source_component=record.source_component,
                status=record.status,
                payload_json=record.payload,
                previous_hash=record.previous_hash,
                hash=record.hash
            )
            self.session.add(db_obj)
            self.session.flush()
        except SQLAlchemyError as e:
            raise PersistenceWriteError(f"Failed to save AuditEvent: {str(e)}")

    def get_by_id(self, audit_id: str) -> Optional[AuditEventRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyAuditEvent).filter_by(audit_id=audit_id).first()
            if not db_obj:
                return None
            return AuditEventRecord(
                audit_id=db_obj.audit_id,
                session_id=db_obj.session_id,
                cycle_id=db_obj.cycle_id,
                event_type=db_obj.event_type,
                entity_type=db_obj.entity_type,
                entity_id=db_obj.entity_id,
                symbol=db_obj.symbol,
                timestamp=db_obj.timestamp.isoformat(),
                source_component=db_obj.source_component,
                status=db_obj.status,
                payload=db_obj.payload_json,
                previous_hash=db_obj.previous_hash,
                hash=db_obj.hash
            )
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read AuditEvent: {str(e)}")

    def get_latest_event(self, session_id: str) -> Optional[AuditEventRecord]:
        try:
            db_obj = self.session.query(SQLAlchemyAuditEvent)\
                .filter_by(session_id=session_id)\
                .order_by(SQLAlchemyAuditEvent.timestamp.desc())\
                .first()
            if not db_obj:
                return None
            return self.get_by_id(db_obj.audit_id)
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read latest AuditEvent: {str(e)}")

    def get_chain(self, session_id: str) -> List[AuditEventRecord]:
        try:
            results = self.session.query(SQLAlchemyAuditEvent)\
                .filter_by(session_id=session_id)\
                .order_by(SQLAlchemyAuditEvent.timestamp.asc())\
                .all()
            return [self.get_by_id(r.audit_id) for r in results]
        except SQLAlchemyError as e:
            raise PersistenceReadError(f"Failed to read AuditEvent chain: {str(e)}")
