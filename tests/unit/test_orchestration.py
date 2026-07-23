import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional, cast, Tuple

# Orchestration models
from backend.orchestration.models import TradingCycleInput, TradingCycleResult, TradingCycleStatus
from backend.orchestration.orchestrator import TradingCycleOrchestrator
from backend.orchestration.policy import TradingCyclePolicy
from backend.orchestration.exceptions import OrchestrationError, LineageIntegrityError

# Engine models
from backend.decision.models import MLSignal, FusionResult, TradeProposal, IntelligenceSnapshot
from backend.risk.models import RiskAuthorizationResult, RiskAuthorizationStatus, RiskContext
from backend.positioning.models import PositionSizeResult, PositionSizingContext
from backend.execution_authorization.models import (
    ExecutionAuthorizationResult,
    ExecutionAuthorizationStatus,
    OrderIntent,
    ExecutionContext,
    ExecutionEnvironment,
    OrderDirection,
    OrderType
)
from backend.execution_adapter.models import ExecutionResult, ExecutionStatus, Fill, PaperExecutionContext
from backend.portfolio.models import PortfolioState, Position, PositionSide
from backend.position_lifecycle.models import (
    PositionLifecycleStatus,
    ExitProposal,
    ProtectivePositionState,
    ProtectiveTriggerType,
    ExitReason
)

# Mocks definition
class MockDecisionFusionEngine:
    def __init__(self, return_none=False, reject=False, rejection_reason="low_confidence", raise_error=False):
        self.return_none = return_none
        self.reject = reject
        self.rejection_reason = rejection_reason
        self.raise_error = raise_error

    def fuse(
        self,
        ml_signal: MLSignal,
        intelligence_snapshot: Optional[IntelligenceSnapshot],
        market_context: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[FusionResult, Optional[TradeProposal]]:
        if self.raise_error:
            raise RuntimeError("Decision Fusion Crash")
        metadata = {}
        if self.reject:
            metadata["rejection_reason"] = self.rejection_reason
        
        fusion_result = FusionResult(
            fusion_id="fusion-123",
            symbol=ml_signal.symbol,
            timeframe=ml_signal.timeframe,
            direction=ml_signal.direction,
            confidence=ml_signal.confidence,
            fusion_score=0.9,
            agreement_score=0.8,
            ml_contribution=0.5,
            intelligence_contribution=0.5,
            intelligence_used=True,
            intelligence_age_seconds=10.0,
            policy_version="fusion-policy-1",
            source_model_version=ml_signal.model_version,
            reasoning_request_id="req-123",
            risk_flags=[],
            timestamp=ml_signal.timestamp,
            metadata=metadata
        )
        
        if self.return_none or self.reject:
            return fusion_result, None
            
        proposal = TradeProposal(
            proposal_id="proposal-123",
            symbol=fusion_result.symbol,
            direction=fusion_result.direction,
            confidence=fusion_result.confidence,
            fusion_score=fusion_result.fusion_score,
            source_model_version=fusion_result.source_model_version,
            fusion_policy_version=fusion_result.policy_version,
            reasoning_request_id=fusion_result.reasoning_request_id,
            created_at=fusion_result.timestamp,
            expires_at=fusion_result.timestamp,
            risk_flags=fusion_result.risk_flags,
            metadata={}
        )
        return fusion_result, proposal

class MockRiskGuardEngine:
    def __init__(self, status=RiskAuthorizationStatus.APPROVED, raise_error=False, risk_fraction=0.01):
        self.status = status
        self.raise_error = raise_error
        self.risk_fraction = risk_fraction

    def evaluate(self, proposal: TradeProposal, context: RiskContext, current_time: datetime) -> RiskAuthorizationResult:
        if self.raise_error:
            raise RuntimeError("Risk Guard Crash")
        return RiskAuthorizationResult(
            authorization_id="risk-456",
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            direction=proposal.direction,
            status=self.status,
            original_confidence=proposal.confidence,
            effective_confidence=proposal.confidence,
            rejection_reasons=["Risk Rejected"] if self.status == RiskAuthorizationStatus.REJECTED else [],
            adjustment_reasons=[],
            triggered_rules=[],
            policy_version="risk-policy-1",
            source_model_version=proposal.source_model_version,
            fusion_policy_version=proposal.fusion_policy_version,
            proposal_created_at=proposal.created_at,
            evaluated_at=proposal.created_at,
            latency_ms=1.0,
            requested_risk_fraction=float(self.risk_fraction),
            authorized_risk_fraction=float(self.risk_fraction)
        )

class MockPositionSizingEngine:
    def __init__(self, quantity=0.1, authorized_risk_fraction=0.01, raise_error=False):
        self.quantity = quantity
        self.authorized_risk_fraction = authorized_risk_fraction
        self.raise_error = raise_error

    def evaluate(self, proposal: TradeProposal, authorization: RiskAuthorizationResult, context: PositionSizingContext) -> PositionSizeResult:
        if self.raise_error:
            raise RuntimeError("Position Sizing Crash")
        return PositionSizeResult(
            sizing_id="sizing-789",
            authorization_id=authorization.authorization_id,
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            direction=proposal.direction,
            quantity=self.quantity,
            position_notional=5000.0,
            entry_price=50000.0,
            stop_loss_price=49000.0 if proposal.direction == "BULLISH" else 51000.0,
            stop_distance_absolute=1000.0,
            stop_distance_fraction=0.02,
            risk_amount=100.0,
            authorized_risk_fraction=float(self.authorized_risk_fraction),
            leverage=1.0,
            estimated_margin_required=5000.0,
            policy_version="sizing-policy-1",
            created_at=proposal.created_at,
            source_model_version=proposal.source_model_version,
            metadata={}
        )

class MockExecutionAuthorizationEngine:
    def __init__(
        self,
        status=ExecutionAuthorizationStatus.AUTHORIZED,
        mismatch_proposal=False,
        mismatch_risk=False,
        mismatch_sizing=False,
        raise_error=False,
        env=ExecutionEnvironment.PAPER
    ):
        self.status = status
        self.mismatch_proposal = mismatch_proposal
        self.mismatch_risk = mismatch_risk
        self.mismatch_sizing = mismatch_sizing
        self.raise_error = raise_error
        self.env = env

    def evaluate(self, proposal: TradeProposal, risk_auth: RiskAuthorizationResult, size_res: PositionSizeResult, context: ExecutionContext) -> ExecutionAuthorizationResult:
        if self.raise_error:
            raise RuntimeError("Execution Authorization Crash")
        proposal_id = "wrong-proposal" if self.mismatch_proposal else proposal.proposal_id
        risk_auth_id = "wrong-risk" if self.mismatch_risk else risk_auth.authorization_id
        sizing_id = "wrong-sizing" if self.mismatch_sizing else size_res.sizing_id
        
        intent = None
        if self.status == ExecutionAuthorizationStatus.AUTHORIZED:
            intent = OrderIntent(
                intent_id="intent-111",
                idempotency_key="idempotency-111",
                proposal_id=proposal_id,
                risk_authorization_id=risk_auth_id,
                sizing_id=sizing_id,
                symbol=proposal.symbol,
                direction=OrderDirection.BUY if proposal.direction == "BULLISH" else OrderDirection.SELL,
                quantity=size_res.quantity,
                order_type=OrderType.MARKET,
                limit_price=None,
                stop_loss=size_res.stop_loss_price,
                take_profit=52000.0 if proposal.direction == "BULLISH" else 48000.0,
                environment=self.env,
                source_model_version=proposal.source_model_version,
                fusion_policy_version=proposal.fusion_policy_version,
                risk_policy_version=risk_auth.policy_version,
                position_sizing_policy_version=size_res.policy_version,
                execution_policy_version="exec-policy-1",
                reasoning_request_id=proposal.reasoning_request_id,
                created_at=proposal.created_at,
                expires_at=proposal.created_at,
                metadata={}
            )
        return ExecutionAuthorizationResult(
            authorization_id="exec-auth-uuid",
            status=self.status,
            intent=intent,
            rejection_reason="Execution Authorization Rejected" if self.status == ExecutionAuthorizationStatus.REJECTED else "",
            triggered_rules=[] if self.status == ExecutionAuthorizationStatus.AUTHORIZED else ["REJECTED_RULE"],
            policy_version="exec-policy-1",
            proposal_id=proposal_id,
            risk_authorization_id=risk_auth_id,
            sizing_id=sizing_id,
            latency_ms=1.0,
            timestamp=proposal.created_at,
            metadata={}
        )

    def authorize_exit(self, proposal: ExitProposal, context: ExecutionContext, risk_policy_version: str, sizing_policy_version: str, idempotency_key: str) -> ExecutionAuthorizationResult:
        if self.raise_error:
            raise RuntimeError("Exit Authorization Crash")
        intent = None
        if self.status == ExecutionAuthorizationStatus.AUTHORIZED:
            intent = OrderIntent(
                intent_id="intent-exit-111",
                idempotency_key=idempotency_key,
                proposal_id=proposal.exit_proposal_id,
                risk_authorization_id="exit-risk-auth",
                sizing_id="exit-sizing-auth",
                symbol=proposal.symbol,
                direction=proposal.exit_direction,
                quantity=float(proposal.requested_quantity),
                order_type=OrderType.MARKET,
                limit_price=None,
                stop_loss=None,
                take_profit=None,
                environment=self.env,
                source_model_version="exit",
                fusion_policy_version="exit",
                risk_policy_version=risk_policy_version,
                position_sizing_policy_version=sizing_policy_version,
                execution_policy_version="exec-exit-v1",
                reasoning_request_id=None,
                created_at=proposal.created_at,
                expires_at=proposal.expires_at,
                metadata={}
            )
        return ExecutionAuthorizationResult(
            authorization_id="exec-exit-auth-uuid",
            status=self.status,
            intent=intent,
            rejection_reason="Exit Authorization Rejected" if self.status == ExecutionAuthorizationStatus.REJECTED else "",
            triggered_rules=[] if self.status == ExecutionAuthorizationStatus.AUTHORIZED else ["EXIT_REJECTED"],
            policy_version="exec-exit-v1",
            proposal_id=proposal.exit_proposal_id,
            risk_authorization_id="exit-risk-auth",
            sizing_id="exit-sizing-auth",
            latency_ms=1.0,
            timestamp=proposal.created_at,
            metadata={}
        )

class MockPaperExecutionAdapter:
    def __init__(self, status=ExecutionStatus.FILLED, fill_qty=None, raise_error=False):
        self.status = status
        self.fill_qty = fill_qty
        self.raise_error = raise_error

    def execute(self, intent: OrderIntent, context: Any) -> ExecutionResult:
        if self.raise_error:
            raise RuntimeError("Execution Adapter Crash")
        filled = self.fill_qty if self.fill_qty is not None else intent.quantity
        fills = [
            Fill(
                fill_id="fill-999",
                intent_id=intent.intent_id,
                symbol=intent.symbol,
                direction=intent.direction,
                quantity=filled,
                price=50000.0,
                notional=filled * 50000.0,
                fee=1.0,
                slippage_amount=0.0,
                timestamp=intent.created_at
            )
        ] if filled > 0 else []
        return ExecutionResult(
            execution_id="exec-888",
            intent_id=intent.intent_id,
            proposal_id=intent.proposal_id,
            risk_authorization_id=intent.risk_authorization_id,
            sizing_id=intent.sizing_id,
            symbol=intent.symbol,
            direction=intent.direction,
            requested_quantity=intent.quantity,
            filled_quantity=filled,
            average_fill_price=50000.0,
            total_notional=filled * 50000.0,
            total_fees=1.0,
            total_slippage=0.0,
            status=self.status,
            fills=fills,
            rejection_reason="Execution Adapter Rejection" if self.status == ExecutionStatus.REJECTED else "",
            adapter_name="MockPaperAdapter",
            environment=intent.environment,
            started_at=intent.created_at,
            completed_at=intent.created_at,
            latency_ms=1.0,
            policy_version=intent.execution_policy_version
        )

class MockPortfolioEngine:
    def __init__(self, raise_error=False):
        self.raise_error = raise_error

    def apply_execution_result(self, result: ExecutionResult) -> PortfolioState:
        if self.raise_error:
            raise RuntimeError("Portfolio Engine Crash")
        positions = {}
        if result.filled_quantity > 0:
            positions[result.symbol] = Position(
                position_id="pos-uuid-111",
                symbol=result.symbol,
                side=PositionSide.LONG if result.direction == OrderDirection.BUY else PositionSide.SHORT,
                quantity=Decimal(str(result.filled_quantity)),
                average_entry_price=Decimal(str(result.average_fill_price)),
                current_price=Decimal(str(result.average_fill_price)),
                position_notional=Decimal(str(result.total_notional)),
                unrealized_pnl=Decimal("0.0"),
                realized_pnl=Decimal("0.0"),
                accumulated_fees=Decimal(str(result.total_fees)),
                leverage=Decimal("1.0"),
                margin_used=Decimal(str(result.total_notional)),
                opened_at=result.started_at,
                updated_at=result.completed_at,
                source_execution_ids=[result.execution_id],
                source_fill_ids=[f.fill_id for f in result.fills]
            )
        return PortfolioState(
            portfolio_id="portfolio-123",
            initial_balance=Decimal("100000.0"),
            cash_balance=Decimal("95000.0"),
            equity=Decimal("100000.0"),
            realized_pnl=Decimal("0.0"),
            unrealized_pnl=Decimal("0.0"),
            total_fees=Decimal(str(result.total_fees)),
            used_margin=Decimal(str(result.total_notional)),
            available_balance=Decimal("95000.0"),
            gross_exposure=Decimal(str(result.total_notional)),
            net_exposure=Decimal(str(result.total_notional)),
            open_position_count=len(positions),
            positions=positions,
            timestamp=result.completed_at
        )

class MockPositionLifecycleStore:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def update(self, state):
        self.data[state.position_id] = state

class MockPositionLifecycleEngine:
    def __init__(self, exit_proposal=None, raise_error=False, raise_sync_error=False):
        self.store = MockPositionLifecycleStore()
        self.exit_proposal = exit_proposal
        self.raise_error = raise_error
        self.raise_sync_error = raise_sync_error
        self.registered_calls = []
        self.synchronized_calls = []

    def evaluate(self, position_id: str, market_price: Decimal, market_timestamp: str, system_timestamp: str) -> Optional[ExitProposal]:
        if self.raise_error:
            raise RuntimeError("Position Lifecycle Evaluation Crash")
        return self.exit_proposal

    def register_position(
        self, position_id, symbol, side, quantity, average_entry_price,
        stop_loss, take_profit, trailing_stop_enabled, trailing_distance,
        trailing_activation_price, timestamp, source_proposal_id, source_execution_id, metadata
    ):
        if self.raise_error:
            raise RuntimeError("Position Lifecycle Registration Crash")
        self.registered_calls.append({
            "position_id": position_id, "symbol": symbol, "side": side, "quantity": quantity,
            "average_entry_price": average_entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
            "trailing_stop_enabled": trailing_stop_enabled, "trailing_distance": trailing_distance,
            "trailing_activation_price": trailing_activation_price, "timestamp": timestamp,
            "source_proposal_id": source_proposal_id, "source_execution_id": source_execution_id, "metadata": metadata
        })
        self.store.update(
            ProtectivePositionState(
                lifecycle_id="lifecycle-uuid",
                position_id=position_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                average_entry_price=average_entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_enabled=trailing_stop_enabled,
                trailing_distance=trailing_distance,
                trailing_activation_price=trailing_activation_price,
                highest_price_since_entry=average_entry_price,
                lowest_price_since_entry=average_entry_price,
                active_trailing_stop_price=None,
                status=PositionLifecycleStatus.OPEN,
                created_at=timestamp,
                updated_at=timestamp,
                policy_version="lifecycle-policy-1",
                source_proposal_id=source_proposal_id,
                source_execution_id=source_execution_id,
                metadata=metadata
            )
        )

    def synchronize_position(self, position_id, current_quantity, timestamp):
        if self.raise_sync_error:
            raise RuntimeError("Position Lifecycle Synchronization Crash")
        self.synchronized_calls.append({
            "position_id": position_id,
            "current_quantity": current_quantity,
            "timestamp": timestamp
        })
        pos = self.store.get(position_id)
        if pos:
            from dataclasses import replace
            self.store.update(replace(pos, quantity=current_quantity))

# Test fixtures helper
def create_valid_input(timestamp=None, sys_timestamp=None) -> TradingCycleInput:
    now_iso = datetime.now(timezone.utc).isoformat()
    t = timestamp or now_iso
    st = sys_timestamp or now_iso
    
    risk_ctx = RiskContext(
        symbol="BTC/USDT",
        timestamp=t,
        equity=100000.0,
        available_balance=100000.0,
        daily_realized_pnl=0.0,
        daily_unrealized_pnl=0.0,
        current_drawdown_pct=0.0,
        portfolio_exposure_pct=0.0,
        symbol_exposure_pct=0.0,
        current_leverage=1.0,
        open_positions_count=0,
        symbol_open_positions_count=0,
        volatility_state="NORMAL",
        consecutive_losses=0,
        market_liquidity_state="NORMAL",
        metadata={}
    )
    
    sizing_ctx = PositionSizingContext(
        symbol="BTC/USDT",
        instrument_type="crypto",
        equity=100000.0,
        available_balance=100000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        market_price=50000.0,
        leverage=1.0,
        contract_size=1.0,
        lot_size=0.001,
        min_quantity=0.001,
        max_quantity=10.0,
        quantity_step=0.001,
        price_tick=0.01,
        current_symbol_exposure=0.0,
        current_portfolio_exposure=0.0,
        market_timestamp=t,
        timestamp=t,
        metadata={}
    )
    
    paper_ctx = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=10.0,
        timestamp=t,
        metadata={}
    )
    
    return TradingCycleInput(
        ml_signal=MLSignal(
            model_version="ml-1",
            symbol="BTC/USDT",
            timeframe="1h",
            prediction=1.0,
            direction="BULLISH",
            confidence=0.8,
            calibrated=True,
            timestamp=t,
            drift_status="normal"
        ),
        risk_context=risk_ctx,
        position_sizing_context=sizing_ctx,
        execution_context=ExecutionContext(
            environment=ExecutionEnvironment.PAPER,
            current_timestamp=t,
            market_timestamp=t,
            execution_enabled=True,
            kill_switch_active=False,
            symbol_trading_enabled=True,
            available_balance=100000.0,
            current_price=50000.0
        ),
        paper_execution_context=paper_ctx,
        timestamp=t,
        metadata={}
    )

def create_orchestrator(
    df=None, rg=None, ps=None, ea=None, exa=None, pe=None, pf=None, pl=None, policy=None
) -> TradingCycleOrchestrator:
    p = policy or TradingCyclePolicy(
        policy_version="test-policy",
        maximum_clock_skew_seconds=5.0,
        maximum_cycle_age_seconds=10.0
    )
    # Duck-typing checks are bypassed via cast to target types
    return TradingCycleOrchestrator(
        policy=p,
        decision_fusion_engine=cast(Any, df or MockDecisionFusionEngine()),
        risk_guard_engine=cast(Any, rg or MockRiskGuardEngine()),
        position_sizing_engine=cast(Any, ps or MockPositionSizingEngine()),
        execution_authorization_engine=cast(Any, ea or MockExecutionAuthorizationEngine()),
        exit_authorization_engine=cast(Any, exa or MockExecutionAuthorizationEngine()),
        paper_execution_adapter=cast(Any, pe or MockPaperExecutionAdapter()),
        portfolio_engine=cast(Any, pf or MockPortfolioEngine()),
        position_lifecycle_engine=cast(Any, pl or MockPositionLifecycleEngine())
    )


# ==================== VERIFICATION SCENARIOS ====================

# 1. Successful standard cycle
def test_successful_trading_cycle():
    orch = create_orchestrator()
    inp = create_valid_input()
    res = orch.run_cycle(inp)
    
    assert res.status == TradingCycleStatus.COMPLETED
    assert res.proposal_generated is True
    assert res.risk_authorized is True
    assert res.execution_authorized is True
    assert res.executed is True
    assert res.portfolio_updated is True
    assert res.lifecycle_registered is True
    assert res.failed_stage is None
    assert res.total_latency_ms > 0
    assert "validation" in res.stage_timings
    assert "fusion" in res.stage_timings
    assert "risk" in res.stage_timings
    assert "sizing" in res.stage_timings
    assert "execution_authorization" in res.stage_timings
    assert "execution" in res.stage_timings
    assert "portfolio" in res.stage_timings
    assert "lifecycle" in res.stage_timings

# 2. Validation failure (Clock skew)
def test_validation_failure_clock_skew():
    from datetime import timedelta
    orch = create_orchestrator()
    future_time = (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()
    inp = create_valid_input(timestamp=future_time)
    res = orch.run_cycle(inp)
    assert res.status == TradingCycleStatus.FAILED
    assert res.failed_stage == "VALIDATION"
    assert res.rejection_reason is not None
    assert "Clock skew" in res.rejection_reason

# 3. Validation failure (Stale cycle data)
def test_validation_failure_stale_cycle():
    from datetime import timedelta
    orch = create_orchestrator()
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
    inp = create_valid_input(timestamp=old_time)
    res = orch.run_cycle(inp)
    assert res.status == TradingCycleStatus.FAILED
    assert res.failed_stage == "VALIDATION"
    assert res.rejection_reason is not None
    assert "stale" in res.rejection_reason.lower()

# 4. Decision Fusion returning None (No proposal)
def test_fusion_no_proposal():
    orch = create_orchestrator(df=MockDecisionFusionEngine(return_none=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.NO_PROPOSAL
    assert res.failed_stage == "FUSION"
    assert "validation" in res.stage_timings
    assert "fusion" in res.stage_timings
    assert "risk" not in res.stage_timings

# 5. Decision Fusion rejection via metadata
def test_fusion_rejection():
    orch = create_orchestrator(df=MockDecisionFusionEngine(reject=True, rejection_reason="drift"))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.FUSION_REJECTED
    assert res.failed_stage == "FUSION"

# 6. Decision Fusion exception crash
def test_fusion_crash():
    orch = create_orchestrator(df=MockDecisionFusionEngine(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.FAILED
    assert res.failed_stage == "FUSION"
    assert res.rejection_reason is not None
    assert "Decision Fusion Crash" in res.rejection_reason

# 7. Risk Guard rejection
def test_risk_rejection():
    orch = create_orchestrator(rg=MockRiskGuardEngine(status=RiskAuthorizationStatus.REJECTED))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.RISK_REJECTED
    assert res.failed_stage == "RISK"
    assert res.risk_authorized is False

# 8. Risk Guard exception crash
def test_risk_crash():
    orch = create_orchestrator(rg=MockRiskGuardEngine(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.FAILED
    assert res.failed_stage == "RISK"

# 9. Position Sizing non-positive quantity
def test_sizing_non_positive_quantity():
    orch = create_orchestrator(ps=MockPositionSizingEngine(quantity=-0.5))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.SIZING_REJECTED
    assert res.failed_stage == "SIZING"

# 10. Position Sizing risk fraction cap violation
def test_sizing_risk_cap_violation():
    orch = create_orchestrator(
        rg=MockRiskGuardEngine(risk_fraction=0.01),
        ps=MockPositionSizingEngine(authorized_risk_fraction=0.05)
    )
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.SIZING_REJECTED
    assert res.failed_stage == "SIZING"
    assert res.rejection_reason is not None
    assert "exceeds RiskGuard authorized risk" in res.rejection_reason

# 11. Position Sizing exception crash
def test_sizing_crash():
    orch = create_orchestrator(ps=MockPositionSizingEngine(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.SIZING_REJECTED
    assert res.failed_stage == "SIZING"

# 12. Execution Authorization rejection
def test_exec_auth_rejection():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(status=ExecutionAuthorizationStatus.REJECTED))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
    assert res.failed_stage == "EXECUTION_AUTHORIZATION"

# 13. Execution Authorization lineage proposal mismatch
def test_exec_auth_proposal_mismatch():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(mismatch_proposal=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
    assert res.failed_stage == "EXECUTION_AUTHORIZATION"
    assert res.rejection_reason is not None
    assert "proposal UUID mismatch" in res.rejection_reason

# 14. Execution Authorization lineage risk auth mismatch
def test_exec_auth_risk_auth_mismatch():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(mismatch_risk=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
    assert res.failed_stage == "EXECUTION_AUTHORIZATION"
    assert res.rejection_reason is not None
    assert "risk authorization UUID mismatch" in res.rejection_reason

# 15. Execution Authorization lineage sizing mismatch
def test_exec_auth_sizing_mismatch():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(mismatch_sizing=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
    assert res.failed_stage == "EXECUTION_AUTHORIZATION"
    assert res.rejection_reason is not None
    assert "sizing UUID mismatch" in res.rejection_reason

# 16. Execution Authorization exception crash
def test_exec_auth_crash():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
    assert res.failed_stage == "EXECUTION_AUTHORIZATION"

# 17. Paper Execution LIVE environment block
def test_exec_block_live():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(env=ExecutionEnvironment.LIVE))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_FAILED
    assert res.failed_stage == "EXECUTION"
    assert res.rejection_reason is not None
    assert "Live execution is not allowed" in res.rejection_reason

# 18. Paper Execution SHADOW environment redirection
def test_exec_shadow_redirect():
    orch = create_orchestrator(ea=MockExecutionAuthorizationEngine(env=ExecutionEnvironment.SHADOW))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.COMPLETED
    assert res.executed is False

# 19. Paper Execution status cancelled/rejected failure
def test_exec_status_failed():
    orch = create_orchestrator(pe=MockPaperExecutionAdapter(status=ExecutionStatus.REJECTED))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_FAILED
    assert res.failed_stage == "EXECUTION"

# 20. Paper Execution exception crash
def test_exec_crash():
    orch = create_orchestrator(pe=MockPaperExecutionAdapter(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.EXECUTION_FAILED
    assert res.failed_stage == "EXECUTION"

# 21. Portfolio update exception crash
def test_portfolio_crash():
    orch = create_orchestrator(pf=MockPortfolioEngine(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.PORTFOLIO_UPDATE_FAILED
    assert res.failed_stage == "PORTFOLIO"
    assert res.executed is True
    assert res.portfolio_updated is False

# 22. Position Lifecycle registration crash
def test_lifecycle_registration_crash():
    orch = create_orchestrator(pl=MockPositionLifecycleEngine(raise_error=True))
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.LIFECYCLE_REGISTRATION_FAILED
    assert res.failed_stage == "LIFECYCLE"
    assert res.portfolio_updated is True
    assert res.lifecycle_registered is False

# 23. Partial fill execution success
def test_partial_fill_success():
    orch = create_orchestrator(
        ps=MockPositionSizingEngine(quantity=0.5),
        pe=MockPaperExecutionAdapter(status=ExecutionStatus.PARTIALLY_FILLED, fill_qty=0.2)
    )
    res = orch.run_cycle(create_valid_input())
    assert res.status == TradingCycleStatus.COMPLETED
    assert res.executed is True
    
    pl_engine = cast(MockPositionLifecycleEngine, orch.position_lifecycle_engine)
    pos_calls = pl_engine.registered_calls
    assert len(pos_calls) == 1
    assert pos_calls[0]["quantity"] == Decimal("0.2")

# 24. Successful exit cycle
def test_successful_exit_cycle():
    pl_engine = MockPositionLifecycleEngine(
        exit_proposal=ExitProposal(
            exit_proposal_id="proposal-exit-999",
            lifecycle_id="lifecycle-uuid",
            position_id="pos-111",
            symbol="BTC/USDT",
            position_side=PositionSide.LONG,
            exit_direction=OrderDirection.SELL,
            requested_quantity=Decimal("1.0"),
            trigger_type=ProtectiveTriggerType.STOP_LOSS,
            exit_reason=ExitReason.STOP_LOSS_TRIGGERED,
            trigger_price=Decimal("49000.0"),
            market_price=Decimal("48950.0"),
            source_stop_loss=Decimal("49000.0"),
            source_take_profit=None,
            source_trailing_stop=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=datetime.now(timezone.utc).isoformat(),
            lifecycle_policy_version="exit-v1",
            source_execution_id="exec-888"
        )
    )
    
    pl_engine.register_position(
        position_id="pos-111",
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        stop_loss=Decimal("49000.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_proposal_id="proposal-123",
        source_execution_id="exec-888",
        metadata={}
    )
    
    orch = create_orchestrator(pl=pl_engine)
    res = orch.run_exit_cycle(
        position_id="pos-111",
        market_price=48950.0,
        market_timestamp=datetime.now(timezone.utc).isoformat(),
        system_timestamp=datetime.now(timezone.utc).isoformat(),
        execution_context=ExecutionContext(
            environment=ExecutionEnvironment.PAPER,
            current_timestamp=datetime.now(timezone.utc).isoformat(),
            market_timestamp=datetime.now(timezone.utc).isoformat(),
            execution_enabled=True,
            kill_switch_active=False,
            symbol_trading_enabled=True,
            available_balance=100000.0,
            current_price=48950.0
        ),
        paper_execution_context={},
        idempotency_key="exit-idempotency-key"
    )
    
    assert res is not None
    assert res.status == TradingCycleStatus.COMPLETED
    assert res.failed_stage is None
    assert "lifecycle_evaluation" in res.stage_timings
    assert "exit_authorization" in res.stage_timings
    assert "execution" in res.stage_timings
    assert "portfolio" in res.stage_timings
    assert "lifecycle_synchronization" in res.stage_timings

# 25. Exit cycle Lifecycle evaluation crash
def test_exit_cycle_evaluation_crash():
    pl_engine = MockPositionLifecycleEngine(raise_error=True)
    orch = create_orchestrator(pl=pl_engine)
    res = orch.run_exit_cycle(
        position_id="pos-111",
        market_price=48950.0,
        market_timestamp=datetime.now(timezone.utc).isoformat(),
        system_timestamp=datetime.now(timezone.utc).isoformat(),
        execution_context=ExecutionContext(
            environment=ExecutionEnvironment.PAPER,
            current_timestamp=datetime.now(timezone.utc).isoformat(),
            market_timestamp=datetime.now(timezone.utc).isoformat(),
            execution_enabled=True,
            kill_switch_active=False,
            symbol_trading_enabled=True,
            available_balance=100000.0,
            current_price=48950.0
        ),
        paper_execution_context={},
        idempotency_key="exit-idempotency-key"
    )
    assert res is not None
    assert res.status == TradingCycleStatus.FAILED
    assert res.failed_stage == "LIFECYCLE"

# 26. Exit cycle Exit Authorization rejection (revert positional closing state)
def test_exit_cycle_auth_rejection_reverts_state():
    pl_engine = MockPositionLifecycleEngine(
        exit_proposal=ExitProposal(
            exit_proposal_id="proposal-exit-999",
            lifecycle_id="lifecycle-uuid",
            position_id="pos-111",
            symbol="BTC/USDT",
            position_side=PositionSide.LONG,
            exit_direction=OrderDirection.SELL,
            requested_quantity=Decimal("1.0"),
            trigger_type=ProtectiveTriggerType.STOP_LOSS,
            exit_reason=ExitReason.STOP_LOSS_TRIGGERED,
            trigger_price=Decimal("49000.0"),
            market_price=Decimal("48950.0"),
            source_stop_loss=Decimal("49000.0"),
            source_take_profit=None,
            source_trailing_stop=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=datetime.now(timezone.utc).isoformat(),
            lifecycle_policy_version="exit-v1",
            source_execution_id="exec-888"
        )
    )
    
    pl_engine.register_position(
        position_id="pos-111",
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        stop_loss=Decimal("49000.0"),
        take_profit=None,
        trailing_stop_enabled=False,
        trailing_distance=None,
        trailing_activation_price=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_proposal_id="proposal-123",
        source_execution_id="exec-888",
        metadata={}
    )
    pos_state = pl_engine.store.get("pos-111")
    assert pos_state is not None
    from dataclasses import replace
    pl_engine.store.update(replace(pos_state, status=PositionLifecycleStatus.CLOSING))
    
    exa_rejected = MockExecutionAuthorizationEngine(status=ExecutionAuthorizationStatus.REJECTED)
    orch = create_orchestrator(pl=pl_engine, exa=exa_rejected)
    res = orch.run_exit_cycle(
        position_id="pos-111",
        market_price=48950.0,
        market_timestamp=datetime.now(timezone.utc).isoformat(),
        system_timestamp=datetime.now(timezone.utc).isoformat(),
        execution_context=ExecutionContext(
            environment=ExecutionEnvironment.PAPER,
            current_timestamp=datetime.now(timezone.utc).isoformat(),
            market_timestamp=datetime.now(timezone.utc).isoformat(),
            execution_enabled=True,
            kill_switch_active=False,
            symbol_trading_enabled=True,
            available_balance=100000.0,
            current_price=48950.0
        ),
        paper_execution_context={},
        idempotency_key="exit-idempotency-key"
    )
    
    assert res is not None
    assert res.status == TradingCycleStatus.EXECUTION_AUTHORIZATION_REJECTED
    
    pos_state_after = pl_engine.store.get("pos-111")
    assert pos_state_after is not None
    assert pos_state_after.status == PositionLifecycleStatus.OPEN

# 27. Exit cycle Execution failure (revert positional closing state)
def test_exit_cycle_execution_failure_reverts_state():
    pl_engine = MockPositionLifecycleEngine(
        exit_proposal=ExitProposal(
            exit_proposal_id="proposal-exit-999",
            lifecycle_id="lifecycle-uuid",
            position_id="pos-111",
            symbol="BTC/USDT",
            position_side=PositionSide.LONG,
            exit_direction=OrderDirection.SELL,
            requested_quantity=Decimal("1.0"),
            trigger_type=ProtectiveTriggerType.STOP_LOSS,
            exit_reason=ExitReason.STOP_LOSS_TRIGGERED,
            trigger_price=Decimal("49000.0"),
            market_price=Decimal("48950.0"),
            source_stop_loss=Decimal("49000.0"),
            source_take_profit=None,
            source_trailing_stop=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=datetime.now(timezone.utc).isoformat(),
            lifecycle_policy_version="exit-v1",
            source_execution_id="exec-888"
        )
    )
    pl_engine.register_position(
        position_id="pos-111", symbol="BTC/USDT", side=PositionSide.LONG, quantity=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"), stop_loss=Decimal("49000.0"), take_profit=None,
        trailing_stop_enabled=False, trailing_distance=None, trailing_activation_price=None,
        timestamp=datetime.now(timezone.utc).isoformat(), source_proposal_id="proposal-123",
        source_execution_id="exec-888", metadata={}
    )
    pos_state = pl_engine.store.get("pos-111")
    assert pos_state is not None
    from dataclasses import replace
    pl_engine.store.update(replace(pos_state, status=PositionLifecycleStatus.CLOSING))
    
    pe_failed = MockPaperExecutionAdapter(status=ExecutionStatus.REJECTED)
    orch = create_orchestrator(pl=pl_engine, pe=pe_failed)
    res = orch.run_exit_cycle(
        position_id="pos-111",
        market_price=48950.0,
        market_timestamp=datetime.now(timezone.utc).isoformat(),
        system_timestamp=datetime.now(timezone.utc).isoformat(),
        execution_context=ExecutionContext(
            environment=ExecutionEnvironment.PAPER,
            current_timestamp=datetime.now(timezone.utc).isoformat(),
            market_timestamp=datetime.now(timezone.utc).isoformat(),
            execution_enabled=True,
            kill_switch_active=False,
            symbol_trading_enabled=True,
            available_balance=100000.0,
            current_price=48950.0
        ),
        paper_execution_context={},
        idempotency_key="exit-idempotency-key"
    )
    assert res is not None
    assert res.status == TradingCycleStatus.EXECUTION_FAILED
    
    pos_state_after = pl_engine.store.get("pos-111")
    assert pos_state_after is not None
    assert pos_state_after.status == PositionLifecycleStatus.OPEN

# 28. Exit cycle Portfolio update crash
def test_exit_cycle_portfolio_crash():
    pl_engine = MockPositionLifecycleEngine(
        exit_proposal=ExitProposal(
            exit_proposal_id="proposal-exit-999",
            lifecycle_id="lifecycle-uuid",
            position_id="pos-111",
            symbol="BTC/USDT",
            position_side=PositionSide.LONG,
            exit_direction=OrderDirection.SELL,
            requested_quantity=Decimal("1.0"),
            trigger_type=ProtectiveTriggerType.STOP_LOSS,
            exit_reason=ExitReason.STOP_LOSS_TRIGGERED,
            trigger_price=Decimal("49000.0"),
            market_price=Decimal("48950.0"),
            source_stop_loss=Decimal("49000.0"),
            source_take_profit=None,
            source_trailing_stop=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=datetime.now(timezone.utc).isoformat(),
            lifecycle_policy_version="exit-v1",
            source_execution_id="exec-888"
        )
    )
    pl_engine.register_position(
        position_id="pos-111", symbol="BTC/USDT", side=PositionSide.LONG, quantity=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"), stop_loss=Decimal("49000.0"), take_profit=None,
        trailing_stop_enabled=False, trailing_distance=None, trailing_activation_price=None,
        timestamp=datetime.now(timezone.utc).isoformat(), source_proposal_id="proposal-123",
        source_execution_id="exec-888", metadata={}
    )
    
    orch = create_orchestrator(pl=pl_engine, pf=MockPortfolioEngine(raise_error=True))
    res = orch.run_exit_cycle(
        position_id="pos-111",
        market_price=48950.0,
        market_timestamp=datetime.now(timezone.utc).isoformat(),
        system_timestamp=datetime.now(timezone.utc).isoformat(),
        execution_context=ExecutionContext(
            environment=ExecutionEnvironment.PAPER,
            current_timestamp=datetime.now(timezone.utc).isoformat(),
            market_timestamp=datetime.now(timezone.utc).isoformat(),
            execution_enabled=True,
            kill_switch_active=False,
            symbol_trading_enabled=True,
            available_balance=100000.0,
            current_price=48950.0
        ),
        paper_execution_context={},
        idempotency_key="exit-idempotency-key"
    )
    assert res is not None
    assert res.status == TradingCycleStatus.PORTFOLIO_UPDATE_FAILED
    assert res.failed_stage == "PORTFOLIO"
