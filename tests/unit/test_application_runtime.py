import pytest
from decimal import Decimal
from typing import Dict, Any, List
from datetime import datetime, timezone

from backend.application.exceptions import (
    ApplicationConfigurationError,
    ComponentInitializationError,
    ComponentDependencyError,
    SessionInitializationError,
)
from backend.application.policy import IntegratedRuntimePolicy
from backend.application.container import QuantForgeContainer
from backend.application.coordinator import IntegratedPaperTradingCoordinator
from backend.application.models import (
    IntegratedRuntimeStatus,
    CycleTriggerType,
)

# Domain objects
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.adapters.memory import MemoryMarketDataProvider
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.schema import FeatureSchema
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

from backend.decision.models import MLSignal


# ─── Mock Predictor ─────────────────────────────────────────────────────────

def mock_predict(features: List[float]) -> float:
    return 0.6


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def base_policies():
    runtime = IntegratedRuntimePolicy(
        policy_version="1.0",
        paper_only=True,
        enabled_symbols=["BTC/USDT"],
        enabled_timeframes=["1m"]
    )
    mkt = MarketDataPolicy(
        allowed_symbols={"BTC/USDT"}
    )
    feat = FeatureRuntimePolicy(
        policy_version="1.0",
        minimum_history=5,
        staleness_limit_seconds=10.0
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
        reject_on_intelligence_risk_flags=["extreme_volatility", "unreliable_data"],
        proposal_ttl_seconds=60.0
    )
    risk = RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.60,
        maximum_proposal_age_seconds=10.0,
        maximum_daily_loss_fraction=0.03,
        maximum_drawdown_fraction=0.05,
        maximum_portfolio_exposure_fraction=0.30,
        maximum_symbol_exposure_fraction=0.10,
        maximum_leverage=3.0,
        maximum_open_positions=3,
        maximum_symbol_open_positions=1,
        maximum_consecutive_losses=3,
        reject_on_critical_volatility=True,
        reject_on_critical_liquidity=True,
        reject_on_critical_drift=True,
        base_risk_fraction=0.02,
        maximum_risk_fraction=0.05,
        minimum_risk_fraction=0.005,
        blocking_risk_flags={"highly_unreliable", "illegal_arbitrage"},
        risk_reducing_flags={"medium_divergence": 0.50, "minor_spread": 0.80},
        informational_risk_flags={"low_volume", "weekend_trade"},
        volatility_adjustments={"HIGH": 0.50, "NORMAL": 1.0, "LOW": 1.0},
    )
    sizing = PositionSizingPolicy(
        policy_version="1.0.0",
        minimum_position_notional=10.0,
        maximum_position_notional=50000.0,
        minimum_quantity=0.001,
        maximum_quantity=100.0,
        maximum_leverage=10.0,
        maximum_margin_fraction=1.0,
        maximum_symbol_exposure_fraction=1.0,
        maximum_portfolio_exposure_fraction=1.0,
        rounding_mode="ROUND",
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
        "runtime": runtime,
        "mkt": mkt,
        "feat": feat,
        "schema": schema,
        "fusion": fusion,
        "risk": risk,
        "sizing": sizing,
        "exec": exec_pol,
        "paper_exec": paper_exec,
        "portfolio": portfolio,
        "lifecycle": lifecycle,
        "cycle": cycle,
        "persist": persist
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_policy_enforces_paper_only():
    with pytest.raises(ApplicationConfigurationError, match="disallowed"):
        IntegratedRuntimePolicy(
            policy_version="1.0",
            paper_only=False,  # Attempting live trading
            enabled_symbols=["BTC/USDT"],
            enabled_timeframes=["1m"]
        )


def test_container_fails_live_environment(base_policies):
    # Alter execution policy to live
    bad_exec = ExecutionPolicy(
        policy_version="1.0",
        allowed_environments=[ExecutionEnvironment.LIVE],
        maximum_market_data_age_seconds=10.0,
        order_intent_ttl_seconds=60.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET]
    )
    provider = MemoryMarketDataProvider()
    
    with pytest.raises(ApplicationConfigurationError, match="LIVE ExecutionEnvironment"):
        QuantForgeContainer(
            runtime_policy=base_policies["runtime"],
            market_data_policy=base_policies["mkt"],
            feature_policy=base_policies["feat"],
            feature_schema=base_policies["schema"],
            predict_fn=mock_predict,
            fusion_policy=base_policies["fusion"],
            risk_policy=base_policies["risk"],
            sizing_policy=base_policies["sizing"],
            execution_policy=bad_exec,
            paper_exec_policy=base_policies["paper_exec"],
            portfolio_policy=base_policies["portfolio"],
            lifecycle_policy=base_policies["lifecycle"],
            cycle_policy=base_policies["cycle"],
            persistence_policy=base_policies["persist"],
            market_data_provider=provider
        )


def test_coordinator_lifecycle(base_policies):
    provider = MemoryMarketDataProvider()
    container = QuantForgeContainer(
        runtime_policy=base_policies["runtime"],
        market_data_policy=base_policies["mkt"],
        feature_policy=base_policies["feat"],
        feature_schema=base_policies["schema"],
        predict_fn=mock_predict,
        fusion_policy=base_policies["fusion"],
        risk_policy=base_policies["risk"],
        sizing_policy=base_policies["sizing"],
        execution_policy=base_policies["exec"],
        paper_exec_policy=base_policies["paper_exec"],
        portfolio_policy=base_policies["portfolio"],
        lifecycle_policy=base_policies["lifecycle"],
        cycle_policy=base_policies["cycle"],
        persistence_policy=base_policies["persist"],
        market_data_provider=provider
    )
    
    coordinator = IntegratedPaperTradingCoordinator(container)
    assert coordinator.status == IntegratedRuntimeStatus.INITIALIZED
    
    # Pre-populate some dummy tickers and books to pass health pre-flight
    from backend.market_data.models import TickerSnapshot, OrderBookSnapshot
    now_str = datetime.now(timezone.utc).isoformat()
    ticker = TickerSnapshot(
        symbol="BTCUSDT",
        bid=Decimal("50000"),
        ask=Decimal("50010"),
        last=Decimal("50005"),
        bid_quantity=Decimal("1"),
        ask_quantity=Decimal("1"),
        volume_24h=Decimal("100"),
        timestamp=now_str,
        source="memory",
        received_at=now_str
    )
    book = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[],
        asks=[],
        sequence=1,
        timestamp=now_str,
        source="memory",
        received_at=now_str
    )
    container.market_store.update_ticker(ticker)
    container.market_store.update_order_book(book)

    coordinator.start_session()
    assert coordinator.status == IntegratedRuntimeStatus.WARMING_UP
    
    # Try pausing
    coordinator.pause()
    assert coordinator.status == IntegratedRuntimeStatus.PAUSED
    
    # Try resuming
    coordinator.resume()
    assert coordinator.status == IntegratedRuntimeStatus.WARMING_UP
    
    # Stop session
    summary = coordinator.stop_session("Finished test")
    assert coordinator.status == IntegratedRuntimeStatus.STOPPED
    assert summary.stop_reason == "Finished test"
