import pytest
from decimal import Decimal
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, List

from backend.application.exceptions import (
    ApplicationConfigurationError,
    ComponentInitializationError,
    ComponentDependencyError,
)
from backend.application.policy import IntegratedRuntimePolicy
from backend.application.container import QuantForgeContainer
from backend.application.coordinator import IntegratedPaperTradingCoordinator
from backend.application.models import (
    IntegratedRuntimeStatus,
    CycleTriggerType,
)

# Domain classes
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.adapters.memory import MemoryMarketDataProvider
from backend.market_data.models import TickerSnapshot, OrderBookSnapshot, Candle, MarketDataSnapshot
from backend.market_data.bridge import MarketDataSnapshotUpdated
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.buffer import BufferCandle
from backend.decision.policy import FusionPolicy
from backend.risk.policy import RiskPolicy
from backend.positioning.policy import PositionSizingPolicy
from backend.execution_authorization.policy import ExecutionPolicy
from backend.execution_authorization.models import ExecutionEnvironment, OrderType
from backend.execution_adapter.policy import PaperExecutionPolicy
from backend.portfolio.policy import PortfolioPolicy
from backend.position_lifecycle.policy import PositionLifecyclePolicy
from backend.orchestration.policy import TradingCyclePolicy
from backend.persistence.policy import PersistencePolicy

def mock_predict(features: List[float]) -> float:
    return 0.9  # Generate strong bullish prediction

@pytest.fixture
def policies():
    symbol = "BTCUSDT"
    timeframe = "1m"
    
    runtime = IntegratedRuntimePolicy(
        policy_version="1.0",
        paper_only=True,
        enabled_symbols=[symbol],
        enabled_timeframes=[timeframe]
    )
    mkt = MarketDataPolicy(
        allowed_symbols={symbol}
    )
    feat = FeatureRuntimePolicy(
        policy_version="1.0",
        minimum_history=2,
        staleness_limit_seconds=60.0,
        default_timeframe=timeframe
    )
    schema = FeatureSchema(
        schema_id="test_schema",
        schema_version="1.0",
        feature_names=["f1", "f2"]
    )
    fusion = FusionPolicy(
        policy_version="v1.0",
        ml_weight=0.7,
        intelligence_weight=0.3,
        minimum_ml_confidence=0.5,
        minimum_fusion_confidence=0.5,
        minimum_agreement_score=-1.0,
        allow_ml_only=True,
        reject_on_critical_drift=True,
        reject_on_intelligence_risk_flags=["unreliable_data"],
        proposal_ttl_seconds=60.0
    )
    risk = RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.60,
        maximum_proposal_age_seconds=10.0,
        maximum_daily_loss_fraction=0.03,
        maximum_drawdown_fraction=0.05,
        maximum_portfolio_exposure_fraction=0.30,
        maximum_symbol_exposure_fraction=0.30,
        maximum_leverage=3.0,
        maximum_open_positions=3,
        maximum_symbol_open_positions=1,
        maximum_consecutive_losses=3,
        reject_on_critical_volatility=True,
        reject_on_critical_liquidity=True,
        reject_on_critical_drift=True,
        base_risk_fraction=0.005,
        maximum_risk_fraction=0.05,
        minimum_risk_fraction=0.005,
        blocking_risk_flags={"highly_unreliable"},
        risk_reducing_flags={"medium_divergence": 0.50},
        informational_risk_flags={"low_volume"},
        volatility_adjustments={"HIGH": 0.50, "NORMAL": 1.0, "LOW": 1.0},
    )
    sizing = PositionSizingPolicy(
        policy_version="1.0.0",
        minimum_position_notional=10.0,
        maximum_position_notional=150000.0,
        minimum_quantity=0.001,
        maximum_quantity=100.0,
        maximum_leverage=10.0,
        maximum_margin_fraction=1.0,
        maximum_symbol_exposure_fraction=1.0,
        maximum_portfolio_exposure_fraction=1.0,
        rounding_mode="DOWN",
        reject_if_below_min_quantity=True,
        reject_if_above_max_quantity=True,
        reject_if_stop_distance_invalid=True,
        reject_if_market_data_stale=True,
        market_data_max_age_seconds=10.0,
        authorization_max_age_seconds=10.0,
    )
    exec_pol = ExecutionPolicy(
        policy_version="exec-policy-1.0",
        allowed_environments=[ExecutionEnvironment.PAPER, ExecutionEnvironment.SHADOW],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET, OrderType.LIMIT],
        allow_live_execution_intents=False,
        require_execution_enabled=True,
        reject_when_kill_switch_active=True,
        require_symbol_enabled=True,
        maximum_clock_skew_seconds=5.0
    )
    paper_exec = PaperExecutionPolicy(
        policy_version="exec-policy-v1",
        maximum_market_data_age_seconds=10.0,
        maximum_future_clock_skew_seconds=2.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        allow_partial_fills=True,
        minimum_fill_quantity=0.0001,
        reject_if_insufficient_liquidity=False,
        intent_max_age_seconds=60.0,
        execution_result_ttl_seconds=3600.0,
    )
    portfolio = PortfolioPolicy(
        policy_version="policy-v1",
        supported_instrument_types=["linear_perpetual", "spot"],
        allow_position_reversal=True,
        maximum_open_positions=5,
        maximum_symbol_positions=1,
        maximum_gross_exposure_fraction=Decimal("3.0"),
        maximum_net_exposure_fraction=Decimal("2.5"),
        maximum_leverage=Decimal("20.0"),
        market_price_max_age_seconds=60.0,
        maximum_future_clock_skew_seconds=10.0,
        accounting_tolerance=Decimal("0.001")
    )
    lifecycle = PositionLifecyclePolicy(
        policy_version="1.0.0",
        allow_stop_loss=True,
        require_stop_loss=False,
        allow_take_profit=True,
        require_take_profit=False,
        allow_trailing_stop=True,
        minimum_stop_distance_fraction=Decimal("0.01"),
        maximum_stop_distance_fraction=Decimal("0.20"),
        minimum_take_profit_distance_fraction=Decimal("0.01"),
        trailing_distance_mode="ABSOLUTE",
        minimum_trailing_distance=Decimal("1.0"),
        maximum_trailing_distance=Decimal("100.0"),
        allow_breakeven=True,
        breakeven_activation_fraction=Decimal("0.05"),
        breakeven_offset_fraction=Decimal("0.005")
    )
    cycle = TradingCyclePolicy(
        policy_version="1.0"
    )
    persist = PersistencePolicy(
        persistence_enabled=False
    )
    return {
        "runtime": runtime, "mkt": mkt, "feat": feat, "schema": schema,
        "fusion": fusion, "risk": risk, "sizing": sizing, "exec": exec_pol,
        "paper_exec": paper_exec, "portfolio": portfolio, "lifecycle": lifecycle,
        "cycle": cycle, "persist": persist
    }

def test_integrated_paper_trading_loop(policies):
    from backend.indicator.indicator_engine import IndicatorEngine
    # Patch indicator engine constraints
    old_min = IndicatorEngine.MIN_CANDLES
    old_calc = IndicatorEngine.calculate
    IndicatorEngine.MIN_CANDLES = 2
    IndicatorEngine.calculate = lambda self, candles: {"f1": 1.0, "f2": 2.0}
    
    try:
        symbol = "BTCUSDT"
        timeframe = "1m"
        provider = MemoryMarketDataProvider()
        
        container = QuantForgeContainer(
            runtime_policy=policies["runtime"],
            market_data_policy=policies["mkt"],
            feature_policy=policies["feat"],
            feature_schema=policies["schema"],
            predict_fn=mock_predict,
            fusion_policy=policies["fusion"],
            risk_policy=policies["risk"],
            sizing_policy=policies["sizing"],
            execution_policy=policies["exec"],
            paper_exec_policy=policies["paper_exec"],
            portfolio_policy=policies["portfolio"],
            lifecycle_policy=policies["lifecycle"],
            cycle_policy=policies["cycle"],
            persistence_policy=policies["persist"],
            market_data_provider=provider
        )
        
        coordinator = IntegratedPaperTradingCoordinator(container)
        assert coordinator.status == IntegratedRuntimeStatus.INITIALIZED
        
        # Start session
        coordinator.start_session()
        assert coordinator.status == IntegratedRuntimeStatus.WARMING_UP
        
        def push_market_data(price, step_idx):
            now_iso = datetime.now(timezone.utc).isoformat()
            ticker = TickerSnapshot(
                symbol=symbol,
                bid=Decimal(str(price - 1.0)),
                ask=Decimal(str(price + 1.0)),
                last=Decimal(str(price)),
                bid_quantity=Decimal("10"),
                ask_quantity=Decimal("10"),
                volume_24h=Decimal("1000"),
                timestamp=now_iso,
                source="memory",
                received_at=now_iso
            )
            book = OrderBookSnapshot(
                symbol=symbol,
                bids=[],
                asks=[],
                sequence=1,
                timestamp=now_iso,
                source="memory",
                received_at=now_iso
            )
            container.market_store.update_ticker(ticker)
            container.market_store.update_order_book(book)
            
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=now_iso,
                close_time=now_iso,
                open=Decimal(str(price)),
                high=Decimal(str(price + 2.0)),
                low=Decimal(str(price - 2.0)),
                close=Decimal(str(price)),
                volume=Decimal("10.0"),
                trade_count=100,
                closed=True,
                source="memory",
                sequence=step_idx,
                received_at=now_iso
            )
            
            snap = MarketDataSnapshot(
                symbol=symbol,
                timestamp=now_iso,
                ticker=ticker,
                latest_trade=None,
                candles={timeframe: [candle]},
                order_book=book,
                source_health="CONNECTED",
                data_age=0.0,
                sequence_state={},
                metadata={}
            )
            
            event = MarketDataSnapshotUpdated(
                event_id=str(uuid.uuid4()),
                event_type="MarketDataSnapshotUpdated",
                timestamp=now_iso,
                runtime_id="test-runtime",
                session_id=container.session.session_id,
                cycle_id=None,
                snapshot=snap
            )
            container.event_bus.publish(event)
            
        # Step 1: Push first snapshot -> Warmup phase (not enough candles yet)
        push_market_data(60000.0, 1)
        results1 = coordinator.step(CycleTriggerType.MARKET_UPDATE)
        assert len(results1) == 0
        assert coordinator.status == IntegratedRuntimeStatus.WARMING_UP
        
        # Step 2: Push second snapshot -> Transition to RUNNING & Executed trade
        push_market_data(60100.0, 2)
        results2 = coordinator.step(CycleTriggerType.MARKET_UPDATE)
        assert len(results2) == 1
        assert results2[0].status.value == "COMPLETED"
        assert coordinator.status == IntegratedRuntimeStatus.RUNNING
        
        # Verify a position was successfully opened
        portfolio_state = container.portfolio_engine.get_state()
        assert len(portfolio_state.positions) == 1
        assert symbol in portfolio_state.positions
        
        # Step 3: Push third snapshot -> Redundant entry request is risk-rejected
        push_market_data(60150.0, 3)
        results3 = coordinator.step(CycleTriggerType.MARKET_UPDATE)
        assert len(results3) == 1
        assert results3[0].status.value == "RISK_REJECTED"
        
        # Stop session
        summary = coordinator.stop_session("shutdown")
        assert coordinator.status == IntegratedRuntimeStatus.STOPPED
        assert summary.total_cycles == 3
        
    finally:
        # Revert indicator patches
        IndicatorEngine.MIN_CANDLES = old_min
        IndicatorEngine.calculate = old_calc
