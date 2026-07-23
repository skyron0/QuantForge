import argparse
import sys
import time
import logging
from decimal import Decimal
from datetime import datetime, timezone

from configs.settings import settings
from backend.application.container import QuantForgeContainer
from backend.application.coordinator import IntegratedPaperTradingCoordinator
from backend.application.models import CycleTriggerType
from backend.market_data.adapters.memory import MemoryMarketDataProvider
from backend.market_data.models import TickerSnapshot, OrderBookSnapshot, Candle

# Policy imports
from backend.application.policy import IntegratedRuntimePolicy
from backend.market_data.policy import MarketDataPolicy
from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.schema import FeatureSchema
from backend.decision.policy import FusionPolicy
from backend.risk.policy import RiskPolicy
from backend.positioning.policy import PositionSizingPolicy
from backend.execution_authorization.policy import ExecutionPolicy
from backend.execution_adapter.policy import PaperExecutionPolicy
from backend.portfolio.policy import PortfolioPolicy
from backend.position_lifecycle.policy import PositionLifecyclePolicy
from backend.orchestration.policy import TradingCyclePolicy
from backend.persistence.policy import PersistencePolicy

# Mock predict function for demonstration
def sample_predict(features: list) -> float:
    return 0.9

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout
    )

def create_policies(symbol: str, timeframe: str):
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
        staleness_limit_seconds=settings.FEATURE_STALENESS_LIMIT_SECONDS,
        default_timeframe=timeframe
    )
    schema = FeatureSchema(
        schema_id="demo_schema",
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
        reject_on_intelligence_risk_flags=["extreme_volatility"],
        proposal_ttl_seconds=60.0
    )
    risk = RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.6,
        maximum_proposal_age_seconds=10.0,
        maximum_daily_loss_fraction=0.03,
        maximum_drawdown_fraction=0.05,
        maximum_portfolio_exposure_fraction=0.3,
        maximum_symbol_exposure_fraction=0.1,
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
        risk_reducing_flags={"medium_divergence": 0.5},
        informational_risk_flags={"low_volume"},
        volatility_adjustments={"HIGH": 0.5, "NORMAL": 1.0, "LOW": 1.0}
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
        authorization_max_age_seconds=10.0
    )
    from backend.execution_authorization.models import OrderType
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
        execution_result_ttl_seconds=3600.0
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

def feed_data(container: QuantForgeContainer, symbol: str, timeframe: str, price: float, index: int):
    norm_symbol = symbol.replace("/", "").upper()
    now_iso = datetime.now(timezone.utc).isoformat()
    
    ticker = TickerSnapshot(
        symbol=norm_symbol,
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
        symbol=norm_symbol,
        bids=[],
        asks=[],
        sequence=1,
        timestamp=now_iso,
        source="memory",
        received_at=now_iso
    )
    container.market_store.update_ticker(ticker)
    container.market_store.update_order_book(book)
    
    if symbol != norm_symbol:
        ticker_raw = TickerSnapshot(
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
        book_raw = OrderBookSnapshot(
            symbol=symbol,
            bids=[],
            asks=[],
            sequence=1,
            timestamp=now_iso,
            source="memory",
            received_at=now_iso
        )
        container.market_store.update_ticker(ticker_raw)
        container.market_store.update_order_book(book_raw)

    # Ingest a candle matching Candle model requirements
    candle = Candle(
        symbol=norm_symbol,
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
        sequence=index,
        received_at=now_iso
    )
    # Populate the snapshot structure expectation
    from backend.market_data.models import MarketDataSnapshot
    snapshot = MarketDataSnapshot(
        symbol=norm_symbol,
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
    
    # Broadcast event to trigger features
    import uuid
    from backend.market_data.bridge import MarketDataSnapshotUpdated
    event = MarketDataSnapshotUpdated(
        event_id=str(uuid.uuid4()),
        event_type="MarketDataSnapshotUpdated",
        timestamp=now_iso,
        runtime_id="demo-runtime",
        session_id=container.session.session_id,
        cycle_id=None,
        snapshot=snapshot
    )
    container.event_bus.publish(event)

def main():
    parser = argparse.ArgumentParser(description="QuantForge Developer Paper Trading Session CLI Runner")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to paper trade")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe")
    parser.add_argument("--steps", type=str, default="10", help="Number of simulated steps to loop")
    parser.add_argument("--price", type=float, default=60000.0, help="Initial mock price")
    parser.add_argument("--verbose", action="store_true", help="Enable debug/verbose printing")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    steps = int(args.steps)
    
    logger = logging.getLogger("QuantForge.PaperSessionRunner")
    logger.info(f"Starting paper trading session CLI demonstration for {args.symbol}...")
    
    # Patch IndicatorEngine to allow immediate sandbox pipeline executions
    from backend.indicator.indicator_engine import IndicatorEngine
    IndicatorEngine.MIN_CANDLES = 2
    IndicatorEngine.calculate = lambda self, candles: {"f1": 1.0, "f2": 2.0}
    
    # Create policies and register provider
    pols = create_policies(args.symbol, args.timeframe)
    provider = MemoryMarketDataProvider()
    
    logger.info("Initializing QuantForgeContainer dependencies...")
    container = QuantForgeContainer(
        runtime_policy=pols["runtime"],
        market_data_policy=pols["mkt"],
        feature_policy=pols["feat"],
        feature_schema=pols["schema"],
        predict_fn=sample_predict,
        fusion_policy=pols["fusion"],
        risk_policy=pols["risk"],
        sizing_policy=pols["sizing"],
        execution_policy=pols["exec"],
        paper_exec_policy=pols["paper_exec"],
        portfolio_policy=pols["portfolio"],
        lifecycle_policy=pols["lifecycle"],
        cycle_policy=pols["cycle"],
        persistence_policy=pols["persist"],
        market_data_provider=provider
    )
    
    coordinator = IntegratedPaperTradingCoordinator(container)
    logger.info("Session coordinator initialized. Starting session...")
    coordinator.start_session()
    
    logger.info(f"Stepping trading cycle {steps} times...")
    price = args.price
    for i in range(1, steps + 1):
        # Walk price randomly
        import random
        price += random.choice([-50.0, -10.0, 0.0, 15.0, 60.0])
        
        logger.info(f"--- Step {i}/{steps} (Price: {price:.2f}) ---")
        feed_data(container, args.symbol, args.timeframe, price, i)
        
        # Step coordinator
        results = coordinator.step(CycleTriggerType.MARKET_UPDATE)
        for r in results:
            logger.info(f"Cycle result status: {r.status.value}")
            if r.status.value == "SUCCESS":
                p_state = coordinator.container.portfolio_engine.get_state()
                logger.info(f"Portfolio equity: {p_state.equity} cash: {p_state.available_balance}")
                
        time.sleep(0.5)
        
    logger.info("Simulation complete. Stopping session...")
    summary = coordinator.stop_session("CLI loop iteration finish")
    
    logger.info("==================================================")
    logger.info("SESSION RUN SUMMARY:")
    logger.info(f"Session ID: {summary.session_id}")
    logger.info(f"Total Cycles Executed: {summary.total_cycles}")
    logger.info(f"Realized PnL: {summary.realized_pnl:.2f}")
    logger.info(f"Final Portfolio Equity: {summary.final_equity:.2f}")
    logger.info(f"Realized Fees: {summary.total_fees:.2f}")
    logger.info("==================================================")

if __name__ == "__main__":
    from backend.execution_authorization.models import ExecutionEnvironment
    main()
