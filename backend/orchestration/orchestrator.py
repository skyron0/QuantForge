import uuid
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Tuple, Dict, Any, List

from backend.orchestration.models import (
    TradingCycleInput,
    TradingCycleResult,
    TradingCycleStatus
)
from backend.orchestration.policy import TradingCyclePolicy
from backend.orchestration.exceptions import (
    OrchestrationValidationError,
    StageExecutionError,
    LineageIntegrityError,
    PortfolioUpdateError,
    LifecycleRegistrationError
)
from backend.orchestration.telemetry import TradingCycleTelemetrySink

# Engine & adapter type hints/imports
from backend.decision.fusion import DecisionFusionEngine
from backend.decision.models import TradeProposal
from backend.risk.guard import RiskGuardEngine
from backend.risk.models import RiskAuthorizationStatus, RiskAuthorizationResult
from backend.positioning.sizing import PositionSizingEngine
from backend.positioning.models import PositionSizeResult
from backend.execution_authorization.authorization import ExecutionAuthorizationEngine
from backend.execution_authorization.models import (
    ExecutionAuthorizationStatus,
    ExecutionAuthorizationResult,
    ExecutionEnvironment,
    OrderIntent,
    ExecutionContext
)
from backend.execution_adapter.paper import PaperExecutionAdapter
from backend.execution_adapter.models import ExecutionStatus, ExecutionResult
from backend.portfolio.portfolio import PortfolioEngine
from backend.portfolio.models import PortfolioState
from backend.position_lifecycle.lifecycle import PositionLifecycleEngine
from backend.position_lifecycle.bridge import ExitAuthorizationEngine
from backend.position_lifecycle.models import ExitProposal, PositionLifecycleStatus


def parse_iso(ts: str) -> datetime:
    if not ts:
        raise OrchestrationValidationError("Empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError as e:
        raise OrchestrationValidationError(f"Invalid timestamp format: {ts}") from e


class TradingCycleOrchestrator:
    """
    Coordinates and sequences the QuantForge deterministic trading cycle stages.
    Enforces absolute short-circuit guarantees on failure/rejection at any stage.
    """
    def __init__(
        self,
        policy: TradingCyclePolicy,
        decision_fusion_engine: DecisionFusionEngine,
        risk_guard_engine: RiskGuardEngine,
        position_sizing_engine: PositionSizingEngine,
        execution_authorization_engine: ExecutionAuthorizationEngine,
        paper_execution_adapter: PaperExecutionAdapter,
        portfolio_engine: PortfolioEngine,
        position_lifecycle_engine: PositionLifecycleEngine,
        exit_authorization_engine: Optional[ExitAuthorizationEngine] = None,
        telemetry_sink: Optional[TradingCycleTelemetrySink] = None
    ) -> None:
        self.policy = policy
        self.decision_fusion_engine = decision_fusion_engine
        self.risk_guard_engine = risk_guard_engine
        self.position_sizing_engine = position_sizing_engine
        self.execution_authorization_engine = execution_authorization_engine
        self.paper_execution_adapter = paper_execution_adapter
        self.portfolio_engine = portfolio_engine
        self.position_lifecycle_engine = position_lifecycle_engine
        
        if exit_authorization_engine is None:
            self.exit_authorization_engine = ExitAuthorizationEngine(
                execution_authorization_engine.policy
            )
        else:
            self.exit_authorization_engine = exit_authorization_engine
            
        self.telemetry_sink = telemetry_sink

    def run_cycle(self, input_data: TradingCycleInput) -> TradingCycleResult:
        """
        Executes a complete entry cycle from MLSignal down to Position Lifecycle registration.
        """
        start_counter = time.perf_counter()
        cycle_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        
        stage_timings: Dict[str, float] = {}

        # 1. Input and clock validation
        validation_start = time.perf_counter()
        try:
            self._validate_input_and_clocks(input_data)
            stage_timings["validation"] = (time.perf_counter() - validation_start) * 1000.0
        except OrchestrationValidationError as e:
            stage_timings["validation"] = (time.perf_counter() - validation_start) * 1000.0
            completed_at = datetime.now(timezone.utc).isoformat()
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return TradingCycleResult(
                cycle_id=cycle_id,
                symbol=input_data.ml_signal.symbol,
                timeframe=input_data.ml_signal.timeframe,
                status=TradingCycleStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                total_latency_ms=latency_ms,
                rejection_stage="VALIDATION",
                failed_stage="VALIDATION",
                rejection_reason=str(e),
                stage_timings=stage_timings,
                policy_version=self.policy.policy_version,
                metadata=input_data.metadata
            )

        # Audit and lineage tracking variables
        fusion_id: Optional[str] = None
        proposal_id: Optional[str] = None
        risk_auth_id: Optional[str] = None
        sizing_id: Optional[str] = None
        execution_auth_id: Optional[str] = None
        intent_id: Optional[str] = None
        execution_id: Optional[str] = None
        fill_ids: List[str] = []

        intel_used = False
        proposal_generated = False
        risk_authorized = False
        execution_authorized = False
        executed = False
        portfolio_updated = False
        lifecycle_registered = False

        proposal: Optional[TradeProposal] = None
        risk_result: Optional[RiskAuthorizationResult] = None
        sizing_result: Optional[PositionSizeResult] = None
        intent: Optional[OrderIntent] = None
        execution_result: Optional[ExecutionResult] = None
        portfolio_state: Optional[PortfolioState] = None

        status = TradingCycleStatus.FAILED
        rejection_stage: Optional[str] = None
        rejection_reason: Optional[str] = None

        curr_dt = parse_iso(input_data.timestamp)

        # 2. Decision Fusion Stage
        fusion_start = time.perf_counter()
        try:
            fusion_result, proposal = self.decision_fusion_engine.fuse(
                ml_signal=input_data.ml_signal,
                intelligence_snapshot=input_data.intelligence_snapshot,
                now=curr_dt
            )
            fusion_id = fusion_result.fusion_id
            intel_used = fusion_result.intelligence_used
            stage_timings["fusion"] = (time.perf_counter() - fusion_start) * 1000.0
            
            if proposal is None:
                proposal_generated = False
                rejection_reason = fusion_result.metadata.get("rejection_reason")
                rejection_stage = "FUSION"
                if rejection_reason and ("confidence" in rejection_reason.lower() or "drift" in rejection_reason.lower() or "allow_ml_only" in rejection_reason.lower()):
                    status = TradingCycleStatus.FUSION_REJECTED
                else:
                    status = TradingCycleStatus.NO_PROPOSAL
                raise StageExecutionError("Decision Fusion produced no trade proposal")
            
            proposal_id = proposal.proposal_id
            proposal_generated = True
        except StageExecutionError:
            pass
        except Exception as e:
            if "fusion" not in stage_timings:
                stage_timings["fusion"] = (time.perf_counter() - fusion_start) * 1000.0
            rejection_stage = "FUSION"
            rejection_reason = str(e)
            status = TradingCycleStatus.FAILED

        # 3. Risk Guard Stage
        if proposal_generated and status == TradingCycleStatus.FAILED:
            risk_start = time.perf_counter()
            try:
                assert proposal is not None
                risk_result = self.risk_guard_engine.evaluate(
                    proposal=proposal,
                    context=input_data.risk_context,
                    current_time=curr_dt
                )
                risk_auth_id = risk_result.authorization_id
                stage_timings["risk"] = (time.perf_counter() - risk_start) * 1000.0
                
                if risk_result.status == RiskAuthorizationStatus.REJECTED:
                    status = TradingCycleStatus.RISK_REJECTED
                    rejection_stage = "RISK"
                    rejection_reason = "; ".join(risk_result.rejection_reasons)
                else:
                    risk_authorized = True
            except Exception as e:
                if "risk" not in stage_timings:
                    stage_timings["risk"] = (time.perf_counter() - risk_start) * 1000.0
                rejection_stage = "RISK"
                rejection_reason = str(e)
                status = TradingCycleStatus.FAILED

        # 4. Position Sizing Stage
        if risk_authorized and status == TradingCycleStatus.FAILED:
            sizing_start = time.perf_counter()
            try:
                assert proposal is not None
                assert risk_result is not None
                sizing_result = self.position_sizing_engine.evaluate(
                    proposal=proposal,
                    authorization=risk_result,
                    context=input_data.position_sizing_context
                )
                sizing_id = sizing_result.sizing_id
                stage_timings["sizing"] = (time.perf_counter() - sizing_start) * 1000.0
                
                if sizing_result.quantity <= 0.0:
                    status = TradingCycleStatus.SIZING_REJECTED
                    rejection_stage = "SIZING"
                    rejection_reason = "Sizing quantity is non-positive"
                # Validate normalized risk fraction cap verification
                elif sizing_result.authorized_risk_fraction > risk_result.authorized_risk_fraction:
                    raise LineageIntegrityError(
                        f"Sizing authorized risk ({sizing_result.authorized_risk_fraction}) "
                        f"exceeds RiskGuard authorized risk ({risk_result.authorized_risk_fraction})"
                    )
            except Exception as e:
                if "sizing" not in stage_timings:
                    stage_timings["sizing"] = (time.perf_counter() - sizing_start) * 1000.0
                status = TradingCycleStatus.SIZING_REJECTED
                rejection_stage = "SIZING"
                rejection_reason = str(e)

        # 5. Execution Authorization Stage
        if risk_authorized and sizing_id and status == TradingCycleStatus.FAILED:
            exec_auth_start = time.perf_counter()
            try:
                assert proposal is not None
                assert risk_result is not None
                assert sizing_result is not None
                exec_auth_result = self.execution_authorization_engine.evaluate(
                    proposal=proposal,
                    risk_auth=risk_result,
                    size_res=sizing_result,
                    context=input_data.execution_context
                )
                execution_auth_id = exec_auth_result.authorization_id
                stage_timings["execution_authorization"] = (time.perf_counter() - exec_auth_start) * 1000.0
                
                if exec_auth_result.status != ExecutionAuthorizationStatus.AUTHORIZED:
                    status = TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
                    rejection_stage = "EXECUTION_AUTHORIZATION"
                    rejection_reason = exec_auth_result.rejection_reason
                else:
                    intent = exec_auth_result.intent
                    assert intent is not None
                    intent_id = intent.intent_id
                    
                    # Lineage validation (Orchestrator defense-in-depth verification)
                    assert proposal is not None
                    assert risk_result is not None
                    assert sizing_result is not None
                    if intent.proposal_id != proposal.proposal_id:
                        raise LineageIntegrityError("Intent proposal UUID mismatch")
                    if intent.risk_authorization_id != risk_result.authorization_id:
                        raise LineageIntegrityError("Intent risk authorization UUID mismatch")
                    if intent.sizing_id != sizing_result.sizing_id:
                        raise LineageIntegrityError("Intent sizing UUID mismatch")
                        
                    execution_authorized = True
            except Exception as e:
                if "execution_authorization" not in stage_timings:
                    stage_timings["execution_authorization"] = (time.perf_counter() - exec_auth_start) * 1000.0
                status = TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
                rejection_stage = "EXECUTION_AUTHORIZATION"
                rejection_reason = str(e)

        # 6. Paper Execution Adapter Stage
        if execution_authorized and status == TradingCycleStatus.FAILED:
            # Shield live environments
            assert intent is not None
            if intent.environment == ExecutionEnvironment.LIVE:
                status = TradingCycleStatus.EXECUTION_FAILED
                rejection_stage = "EXECUTION"
                rejection_reason = "Live execution is not allowed in Sprint 3.7"
            # Support Shadow mode (simulate success without executing)
            elif intent.environment == ExecutionEnvironment.SHADOW:
                # Returns shadow simulated completion without moving positions
                status = TradingCycleStatus.COMPLETED
                executed = False
            else:
                exec_start = time.perf_counter()
                try:
                    assert intent is not None
                    execution_result = self.paper_execution_adapter.execute(
                        intent=intent,
                        context=input_data.paper_execution_context
                    )
                    execution_id = execution_result.execution_id
                    fill_ids = [f.fill_id for f in execution_result.fills]
                    stage_timings["execution"] = (time.perf_counter() - exec_start) * 1000.0
                    
                    if execution_result.status not in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED):
                        status = TradingCycleStatus.EXECUTION_FAILED
                        rejection_stage = "EXECUTION"
                        rejection_reason = f"Execution result: {execution_result.status.value}"
                    else:
                        executed = True
                except Exception as e:
                    if "execution" not in stage_timings:
                        stage_timings["execution"] = (time.perf_counter() - exec_start) * 1000.0
                    status = TradingCycleStatus.EXECUTION_FAILED
                    rejection_stage = "EXECUTION"
                    rejection_reason = str(e)

        # 7. Portfolio Engine Stage (Fills Ingestion)
        if executed and status == TradingCycleStatus.FAILED:
            portfolio_start = time.perf_counter()
            try:
                assert execution_result is not None
                portfolio_state = self.portfolio_engine.apply_execution_result(execution_result)
                portfolio_updated = True
                stage_timings["portfolio"] = (time.perf_counter() - portfolio_start) * 1000.0
            except Exception as e:
                if "portfolio" not in stage_timings:
                    stage_timings["portfolio"] = (time.perf_counter() - portfolio_start) * 1000.0
                status = TradingCycleStatus.PORTFOLIO_UPDATE_FAILED
                rejection_stage = "PORTFOLIO"
                rejection_reason = str(e)
                # Keep lineage details of completed executions (crucial for audit validity)
                completed_at = datetime.now(timezone.utc).isoformat()
                latency_ms = (time.perf_counter() - start_counter) * 1000.0
                res = TradingCycleResult(
                    cycle_id=cycle_id,
                    symbol=input_data.ml_signal.symbol,
                    timeframe=input_data.ml_signal.timeframe,
                    status=status,
                    fusion_id=fusion_id,
                    proposal_id=proposal_id,
                    risk_authorization_id=risk_auth_id,
                    sizing_id=sizing_id,
                    execution_authorization_id=execution_auth_id,
                    intent_id=intent_id,
                    execution_id=execution_id,
                    fill_ids=fill_ids,
                    intelligence_used=intel_used,
                    proposal_generated=proposal_generated,
                    risk_authorized=risk_authorized,
                    execution_authorized=execution_authorized,
                    executed=executed,
                    portfolio_updated=portfolio_updated,
                    lifecycle_registered=lifecycle_registered,
                    rejection_stage=rejection_stage,
                    failed_stage=rejection_stage,
                    rejection_reason=rejection_reason,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_ms=latency_ms,
                    total_latency_ms=latency_ms,
                    stage_timings=stage_timings,
                    policy_version=self.policy.policy_version,
                    metadata=input_data.metadata
                )
                if self.telemetry_sink:
                    self.telemetry_sink.record_cycle(res)
                return res

        # 8. Position Lifecycle Registration Stage
        if portfolio_updated:
            lifecycle_start = time.perf_counter()
            try:
                # Verify that position exists in the portfolio engine
                assert portfolio_state is not None
                pos = portfolio_state.positions.get(input_data.ml_signal.symbol)
                if pos:
                    # Enforce stop-loss and take-profit fields from authorized intent
                    assert intent is not None
                    stop_loss_val = Decimal(str(intent.stop_loss)) if intent.stop_loss is not None else None
                    take_profit_val = Decimal(str(intent.take_profit)) if intent.take_profit is not None else None
                    
                    # Pull trailing parameters from either intent or input metadata
                    trailing_stop_enabled = bool(
                        intent.metadata.get("trailing_stop_enabled") or 
                        input_data.metadata.get("trailing_stop_enabled", False)
                    )
                    
                    trailing_dist_raw = intent.metadata.get("trailing_distance") or input_data.metadata.get("trailing_distance")
                    trailing_distance = Decimal(str(trailing_dist_raw)) if trailing_dist_raw is not None else None
                    
                    trailing_act_raw = intent.metadata.get("trailing_activation_price") or input_data.metadata.get("trailing_activation_price")
                    trailing_activation_price = Decimal(str(trailing_act_raw)) if trailing_act_raw is not None else None

                    # If already registered, synchronize it to account for quantity changes (partial fills)
                    if self.position_lifecycle_engine.store.get(pos.position_id):
                        self.position_lifecycle_engine.synchronize_position(
                            position_id=pos.position_id,
                            current_quantity=pos.quantity,
                            timestamp=input_data.timestamp
                        )
                    else:
                        self.position_lifecycle_engine.register_position(
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            side=pos.side,
                            quantity=pos.quantity,
                            average_entry_price=pos.average_entry_price,
                            stop_loss=stop_loss_val,
                            take_profit=take_profit_val,
                            trailing_stop_enabled=trailing_stop_enabled,
                            trailing_distance=trailing_distance,
                            trailing_activation_price=trailing_activation_price,
                            timestamp=input_data.timestamp,
                            source_proposal_id=proposal_id,
                            source_execution_id=execution_id,
                            metadata=intent.metadata
                        )
                    lifecycle_registered = True
                    status = TradingCycleStatus.COMPLETED
                else:
                    # No active position exists, execution closed it or it reversed
                    status = TradingCycleStatus.COMPLETED
                    lifecycle_registered = False
                stage_timings["lifecycle"] = (time.perf_counter() - lifecycle_start) * 1000.0
            except Exception as e:
                if "lifecycle" not in stage_timings:
                    stage_timings["lifecycle"] = (time.perf_counter() - lifecycle_start) * 1000.0
                # Do NOT rollback or hide executed portfolio stats; fail lifecycle register only
                status = TradingCycleStatus.LIFECYCLE_REGISTRATION_FAILED
                rejection_stage = "LIFECYCLE"
                rejection_reason = str(e)

        completed_at = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.perf_counter() - start_counter) * 1000.0

        res = TradingCycleResult(
            cycle_id=cycle_id,
            symbol=input_data.ml_signal.symbol,
            timeframe=input_data.ml_signal.timeframe,
            status=status,
            fusion_id=fusion_id,
            proposal_id=proposal_id,
            risk_authorization_id=risk_auth_id,
            sizing_id=sizing_id,
            execution_authorization_id=execution_auth_id,
            intent_id=intent_id,
            execution_id=execution_id,
            fill_ids=fill_ids,
            intelligence_used=intel_used,
            proposal_generated=proposal_generated,
            risk_authorized=risk_authorized,
            execution_authorized=execution_authorized,
            executed=executed,
            portfolio_updated=portfolio_updated,
            lifecycle_registered=lifecycle_registered,
            rejection_stage=rejection_stage,
            failed_stage=rejection_stage,
            rejection_reason=rejection_reason,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            total_latency_ms=latency_ms,
            stage_timings=stage_timings,
            policy_version=self.policy.policy_version,
            metadata=input_data.metadata
        )

        if self.telemetry_sink:
            self.telemetry_sink.record_cycle(res)

        return res

    def run_exit_cycle(
        self,
        position_id: str,
        market_price: float,
        market_timestamp: str,
        system_timestamp: str,
        execution_context: ExecutionContext,
        paper_execution_context: Any,
        idempotency_key: str
    ) -> Optional[TradingCycleResult]:
        """
        Coordinates the protective position exit flow when stop triggered by lifecycleengine.
        """
        start_counter = time.perf_counter()
        cycle_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        
        stage_timings: Dict[str, float] = {}

        exit_proposal: Optional[ExitProposal] = None
        exec_auth_result: Optional[ExecutionAuthorizationResult] = None
        intent: Optional[OrderIntent] = None
        execution_result: Optional[ExecutionResult] = None
        portfolio_state: Optional[PortfolioState] = None

        # Step 1: Position Lifecycle Evaluation
        lifecycle_start = time.perf_counter()
        try:
            m_price_dec = Decimal(str(market_price))
            exit_proposal = self.position_lifecycle_engine.evaluate(
                position_id=position_id,
                market_price=m_price_dec,
                market_timestamp=market_timestamp,
                system_timestamp=system_timestamp
            )
            stage_timings["lifecycle_evaluation"] = (time.perf_counter() - lifecycle_start) * 1000.0
        except Exception as e:
            stage_timings["lifecycle_evaluation"] = (time.perf_counter() - lifecycle_start) * 1000.0
            completed_at = datetime.now(timezone.utc).isoformat()
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return TradingCycleResult(
                cycle_id=cycle_id,
                symbol="unknown",
                timeframe="unknown",
                status=TradingCycleStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                total_latency_ms=latency_ms,
                stage_timings=stage_timings,
                rejection_stage="LIFECYCLE",
                failed_stage="LIFECYCLE",
                rejection_reason=str(e),
                policy_version=self.policy.policy_version
            )

        if exit_proposal is None:
            return None

        # Step 2: Exit Authorization via bridge ExitAuthorizationEngine
        exit_auth_start = time.perf_counter()
        try:
            exec_auth_result = self.exit_authorization_engine.authorize_exit(
                proposal=exit_proposal,
                context=execution_context,
                risk_policy_version="exit_engine",
                sizing_policy_version="exit_engine",
                idempotency_key=idempotency_key
            )
            stage_timings["exit_authorization"] = (time.perf_counter() - exit_auth_start) * 1000.0
            
            if exec_auth_result.status != ExecutionAuthorizationStatus.AUTHORIZED:
                # Set underlying lifecycle position from CLOSING back to OPEN to retry exit next tick
                # Fetch position and set it back to open via store
                pos_state = self.position_lifecycle_engine.store.get(position_id)
                if pos_state and pos_state.status == PositionLifecycleStatus.CLOSING:
                    from dataclasses import replace
                    self.position_lifecycle_engine.store.update(
                        replace(pos_state, status=PositionLifecycleStatus.OPEN)
                    )
                
                completed_at = datetime.now(timezone.utc).isoformat()
                latency_ms = (time.perf_counter() - start_counter) * 1000.0
                return TradingCycleResult(
                    cycle_id=cycle_id,
                    symbol=exit_proposal.symbol,
                    timeframe="unknown",
                    status=TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED,
                    proposal_id=exit_proposal.exit_proposal_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_ms=latency_ms,
                    total_latency_ms=latency_ms,
                    stage_timings=stage_timings,
                    rejection_stage="EXECUTION_AUTHORIZATION",
                    failed_stage="EXECUTION_AUTHORIZATION",
                    rejection_reason=exec_auth_result.rejection_reason,
                    policy_version=self.policy.policy_version
                )
        except Exception as e:
            if "exit_authorization" not in stage_timings:
                stage_timings["exit_authorization"] = (time.perf_counter() - exit_auth_start) * 1000.0
            completed_at = datetime.now(timezone.utc).isoformat()
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return TradingCycleResult(
                cycle_id=cycle_id,
                symbol=exit_proposal.symbol,
                timeframe="unknown",
                status=TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED,
                proposal_id=exit_proposal.exit_proposal_id,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                total_latency_ms=latency_ms,
                stage_timings=stage_timings,
                rejection_stage="EXECUTION_AUTHORIZATION",
                failed_stage="EXECUTION_AUTHORIZATION",
                rejection_reason=str(e),
                policy_version=self.policy.policy_version
            )

        intent = exec_auth_result.intent
        assert intent is not None
        
        # Step 3: Paper Execution Adapter
        exec_start = time.perf_counter()
        try:
            execution_result = self.paper_execution_adapter.execute(
                intent=intent,
                context=paper_execution_context
            )
            stage_timings["execution"] = (time.perf_counter() - exec_start) * 1000.0
            
            if execution_result.status not in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED):
                # Set position state back to OPEN on execution failure
                pos_state = self.position_lifecycle_engine.store.get(position_id)
                if pos_state and pos_state.status == PositionLifecycleStatus.CLOSING:
                    from dataclasses import replace
                    self.position_lifecycle_engine.store.update(
                        replace(pos_state, status=PositionLifecycleStatus.OPEN)
                    )
                
                completed_at = datetime.now(timezone.utc).isoformat()
                latency_ms = (time.perf_counter() - start_counter) * 1000.0
                return TradingCycleResult(
                    cycle_id=cycle_id,
                    symbol=exit_proposal.symbol,
                    timeframe="unknown",
                    status=TradingCycleStatus.EXECUTION_FAILED,
                    proposal_id=exit_proposal.exit_proposal_id,
                    intent_id=intent.intent_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_ms=latency_ms,
                    total_latency_ms=latency_ms,
                    stage_timings=stage_timings,
                    rejection_stage="EXECUTION",
                    failed_stage="EXECUTION",
                    rejection_reason=f"Execution result status: {execution_result.status.value}",
                    policy_version=self.policy.policy_version
                )
        except Exception as e:
            if "execution" not in stage_timings:
                stage_timings["execution"] = (time.perf_counter() - exec_start) * 1000.0
            # Set position state back to OPEN on execution adapter crash
            pos_state = self.position_lifecycle_engine.store.get(position_id)
            if pos_state and pos_state.status == PositionLifecycleStatus.CLOSING:
                from dataclasses import replace
                self.position_lifecycle_engine.store.update(
                    replace(pos_state, status=PositionLifecycleStatus.OPEN)
                )

            completed_at = datetime.now(timezone.utc).isoformat()
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return TradingCycleResult(
                cycle_id=cycle_id,
                symbol=exit_proposal.symbol,
                timeframe="unknown",
                status=TradingCycleStatus.EXECUTION_FAILED,
                proposal_id=exit_proposal.exit_proposal_id,
                intent_id=intent.intent_id,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                total_latency_ms=latency_ms,
                stage_timings=stage_timings,
                rejection_stage="EXECUTION",
                failed_stage="EXECUTION",
                rejection_reason=str(e),
                policy_version=self.policy.policy_version
            )

        # Step 4: Portfolio Accounting Update
        portfolio_start = time.perf_counter()
        try:
            portfolio_state = self.portfolio_engine.apply_execution_result(execution_result)
            stage_timings["portfolio"] = (time.perf_counter() - portfolio_start) * 1000.0
        except Exception as e:
            if "portfolio" not in stage_timings:
                stage_timings["portfolio"] = (time.perf_counter() - portfolio_start) * 1000.0
            completed_at = datetime.now(timezone.utc).isoformat()
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return TradingCycleResult(
                cycle_id=cycle_id,
                symbol=exit_proposal.symbol,
                timeframe="unknown",
                status=TradingCycleStatus.PORTFOLIO_UPDATE_FAILED,
                proposal_id=exit_proposal.exit_proposal_id,
                intent_id=intent.intent_id,
                execution_id=execution_result.execution_id,
                fill_ids=[f.fill_id for f in execution_result.fills],
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                total_latency_ms=latency_ms,
                stage_timings=stage_timings,
                rejection_stage="PORTFOLIO",
                failed_stage="PORTFOLIO",
                rejection_reason=str(e),
                policy_version=self.policy.policy_version
            )

        # Step 5: Synchronize Protective State (Close out or adjust quantity)
        lifecycle_sync_start = time.perf_counter()
        try:
            pos = portfolio_state.positions.get(exit_proposal.symbol)
            qty = pos.quantity if pos else Decimal("0")
            self.position_lifecycle_engine.synchronize_position(
                position_id=position_id,
                current_quantity=qty,
                timestamp=system_timestamp
            )
            stage_timings["lifecycle_synchronization"] = (time.perf_counter() - lifecycle_sync_start) * 1000.0
        except Exception as e:
            if "lifecycle_synchronization" not in stage_timings:
                stage_timings["lifecycle_synchronization"] = (time.perf_counter() - lifecycle_sync_start) * 1000.0
            completed_at = datetime.now(timezone.utc).isoformat()
            latency_ms = (time.perf_counter() - start_counter) * 1000.0
            return TradingCycleResult(
                cycle_id=cycle_id,
                symbol=exit_proposal.symbol,
                timeframe="unknown",
                status=TradingCycleStatus.LIFECYCLE_REGISTRATION_FAILED,
                proposal_id=exit_proposal.exit_proposal_id,
                intent_id=intent.intent_id,
                execution_id=execution_result.execution_id,
                fill_ids=[f.fill_id for f in execution_result.fills],
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                total_latency_ms=latency_ms,
                stage_timings=stage_timings,
                rejection_stage="LIFECYCLE",
                failed_stage="LIFECYCLE",
                rejection_reason=str(e),
                policy_version=self.policy.policy_version
            )

        completed_at = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.perf_counter() - start_counter) * 1000.0
        return TradingCycleResult(
            cycle_id=cycle_id,
            symbol=exit_proposal.symbol,
            timeframe="unknown",
            status=TradingCycleStatus.COMPLETED,
            proposal_id=exit_proposal.exit_proposal_id,
            intent_id=intent.intent_id,
            execution_id=execution_result.execution_id,
            fill_ids=[f.fill_id for f in execution_result.fills],
            intelligence_used=False,
            proposal_generated=True,
            risk_authorized=True,
            execution_authorized=True,
            executed=True,
            portfolio_updated=True,
            lifecycle_registered=True,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            total_latency_ms=latency_ms,
            stage_timings=stage_timings,
            policy_version=self.policy.policy_version
        )

    def _validate_input_and_clocks(self, input_data: TradingCycleInput) -> None:
        """Validate input types, age limits, and clock drift skew checks."""
        if not isinstance(input_data, TradingCycleInput):
            raise OrchestrationValidationError("Input data must of type TradingCycleInput")
        if not input_data.ml_signal:
            raise OrchestrationValidationError("MLSignal is required")
        if not input_data.risk_context:
            raise OrchestrationValidationError("RiskContext is required")
        if not input_data.position_sizing_context:
            raise OrchestrationValidationError("PositionSizingContext is required")
        if not input_data.execution_context:
            raise OrchestrationValidationError("ExecutionContext is required")
        if not input_data.paper_execution_context:
            raise OrchestrationValidationError("PaperExecutionContext is required")

        # Time drift checks
        try:
            curr_dt = parse_iso(input_data.timestamp)
        except Exception as e:
            raise OrchestrationValidationError(f"Invalid cycle timestamp: {str(e)}")

        now_utc = datetime.now(timezone.utc)
        
        # Clock skew against system clock
        skew = (curr_dt - now_utc).total_seconds()
        if skew > self.policy.maximum_clock_skew_seconds:
            raise OrchestrationValidationError(
                f"Clock skew check failed: timestamp is {skew}s in future (limit {self.policy.maximum_clock_skew_seconds}s)"
            )
            
        age = (now_utc - curr_dt).total_seconds()
        if age > self.policy.maximum_cycle_age_seconds:
            raise OrchestrationValidationError(
                f"Cycle timestamp is too stale: age is {age}s (limit {self.policy.maximum_cycle_age_seconds}s)"
            )
