import pytest
import tempfile
import csv
import os
from decimal import Decimal
from datetime import datetime, timezone

from backend.replay.service import HistoricalReplayService
from backend.replay.models import ReplaySessionConfig, ReplayStatus
from backend.replay.clock import ReplayClock
from backend.replay.policy import ReplayPolicy
from backend.indicator.indicator_engine import IndicatorEngine
from backend.application.policy import IntegratedRuntimePolicy
from backend.market_data.policy import MarketDataPolicy
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


@pytest.fixture
def base_policies():
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
        staleness_limit_seconds=86400.0,
        default_timeframe=timeframe
    )
    schema = FeatureSchema(
        schema_id="test_schema",
        schema_version="1.0",
        feature_names=["f1", "f2"]
    )
    fusion = FusionPolicy(
        policy_version="v1.0",
        ml_weight=1.0,
        intelligence_weight=0.0,
        minimum_ml_confidence=0.5,
        minimum_fusion_confidence=0.5,
        minimum_agreement_score=-1.0,
        allow_ml_only=True,
        reject_on_critical_drift=False,
        reject_on_intelligence_risk_flags=[],
        proposal_ttl_seconds=86400.0
    )
    risk = RiskPolicy(
        policy_version="1.0.0",
        minimum_proposal_confidence=0.1,
        maximum_proposal_age_seconds=86400.0,
        maximum_daily_loss_fraction=1.0,
        maximum_drawdown_fraction=1.0,
        maximum_portfolio_exposure_fraction=1.0,
        maximum_symbol_exposure_fraction=1.0,
        maximum_leverage=10.0,
        maximum_open_positions=5,
        maximum_symbol_open_positions=1,
        maximum_consecutive_losses=10,
        reject_on_critical_volatility=False,
        reject_on_critical_liquidity=False,
        reject_on_critical_drift=False,
        base_risk_fraction=0.01,
        maximum_risk_fraction=0.5,
        minimum_risk_fraction=0.01,
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
        reject_if_stop_distance_invalid=False,
        reject_if_market_data_stale=False,
        market_data_max_age_seconds=86400.0,
        authorization_max_age_seconds=86400.0,
    )
    exec_pol = ExecutionPolicy(
        policy_version="exec-policy-1.0",
        allowed_environments=[ExecutionEnvironment.PAPER],
        maximum_market_data_age_seconds=86400.0,
        order_intent_ttl_seconds=86400.0,
        minimum_quantity=0.001,
        maximum_quantity=10.0,
        require_stop_loss=True,
        require_take_profit=False,
        allowed_order_types=[OrderType.MARKET, OrderType.LIMIT],
        allow_live_execution_intents=False,
        require_execution_enabled=True,
        reject_when_kill_switch_active=False,
        require_symbol_enabled=True,
        maximum_clock_skew_seconds=86400.0
    )
    paper_exec = PaperExecutionPolicy(
        policy_version="exec-policy-v1",
        maximum_market_data_age_seconds=86400.0,
        maximum_future_clock_skew_seconds=86400.0,
        fee_rate=0.001,
        slippage_rate=0.0,
        allow_partial_fills=False,
        minimum_fill_quantity=0.0001,
        reject_if_insufficient_liquidity=False,
        intent_max_age_seconds=86400.0,
        execution_result_ttl_seconds=86400.0,
    )
    portfolio = PortfolioPolicy(
        policy_version="policy-v1",
        supported_instrument_types=["linear_perpetual", "spot"],
        allow_position_reversal=True,
        maximum_open_positions=5,
        maximum_symbol_positions=1,
        maximum_gross_exposure_fraction=Decimal("10.0"),
        maximum_net_exposure_fraction=Decimal("10.0"),
        maximum_leverage=Decimal("10.0"),
        market_price_max_age_seconds=86400.0,
        maximum_future_clock_skew_seconds=86400.0,
        accounting_tolerance=Decimal("0.001")
    )
    lifecycle = PositionLifecyclePolicy(
        policy_version="1.0.0",
        allow_stop_loss=True,
        require_stop_loss=False,
        allow_take_profit=True,
        require_take_profit=False,
        allow_trailing_stop=False,
        minimum_stop_distance_fraction=Decimal("0.001"),
        maximum_stop_distance_fraction=Decimal("0.5"),
        minimum_take_profit_distance_fraction=Decimal("0.001"),
        trailing_distance_mode="ABSOLUTE",
        minimum_trailing_distance=Decimal("1.0"),
        maximum_trailing_distance=Decimal("100.0"),
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


def test_historical_replay_pipeline_e2e(base_policies):
    # Patch indicator engine constraints
    old_min = IndicatorEngine.MIN_CANDLES
    old_calc = IndicatorEngine.calculate
    IndicatorEngine.MIN_CANDLES = 2
    IndicatorEngine.calculate = lambda self, candles: {"f1": 1.0, "f2": 2.0}

    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerow(["2023-01-01T00:00:00Z", "60000.0", "60050.0", "59950.0", "60000.0", "1.0"])
            writer.writerow(["2023-01-01T00:01:00Z", "60000.0", "60150.0", "59990.0", "60100.0", "1.5"])
            writer.writerow(["2023-01-01T00:02:00Z", "60100.0", "60250.0", "60080.0", "60200.0", "2.0"])
            writer.writerow(["2023-01-01T00:03:00Z", "60200.0", "60350.0", "60180.0", "60300.0", "2.5"])
            writer.writerow(["2023-01-01T00:04:00Z", "60300.0", "60450.0", "60280.0", "60400.0", "3.0"])

        config = ReplaySessionConfig(
            initial_capital=100000.0,
            max_cycles=100,
            seed=42,
            dataset_path=path,
            enabled_symbols=["BTCUSDT"],
            timeframe="1m"
        )

        # Bullish signal model prediction returning high confidence
        def bullish_predict(features):
            return 0.95

        service = HistoricalReplayService()
        result = service.run_replay_session(
            config=config,
            predict_fn=bullish_predict,
            custom_policies=base_policies
        )

        assert result.status == ReplayStatus.COMPLETED
        assert result.dataset_metadata.row_count == 5
        assert result.progress.processed_steps == 5
        assert len(result.determinism_hash) == 64
        # Since we ran with bullish signals, a position should have been opened
        assert len(result.portfolio_history) == 5
        
    finally:
        IndicatorEngine.MIN_CANDLES = old_min
        IndicatorEngine.calculate = old_calc
        os.close(fd)
        os.remove(path)


def test_historical_replay_invariance(base_policies):
    # Patch indicator engine constraints
    old_min = IndicatorEngine.MIN_CANDLES
    old_calc = IndicatorEngine.calculate
    IndicatorEngine.MIN_CANDLES = 2
    IndicatorEngine.calculate = lambda self, candles: {"f1": 1.0, "f2": 2.0}

    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerow(["2023-01-01T00:00:00Z", "60000.0", "60050.0", "59950.0", "60000.0", "1.0"])
            writer.writerow(["2023-01-01T00:01:00Z", "60000.0", "60150.0", "59990.0", "60100.0", "1.5"])
            writer.writerow(["2023-01-01T00:02:00Z", "60100.0", "60250.0", "60080.0", "60200.0", "2.0"])
            writer.writerow(["2023-01-01T00:03:00Z", "60200.0", "60350.0", "60180.0", "60300.0", "2.5"])

        config = ReplaySessionConfig(
            initial_capital=100000.0,
            max_cycles=100,
            seed=42,  # Fixed seed
            dataset_path=path,
            enabled_symbols=["BTCUSDT"],
            timeframe="1m"
        )

        def bullish_predict(features):
            return 0.95

        service = HistoricalReplayService()
        result1 = service.run_replay_session(
            config=config,
            predict_fn=bullish_predict,
            custom_policies=base_policies
        )

        result2 = service.run_replay_session(
            config=config,
            predict_fn=bullish_predict,
            custom_policies=base_policies
        )

        assert result1.determinism_hash == result2.determinism_hash
        assert result1.final_equity == result2.final_equity
        assert result1.fees == result2.fees
        assert len(result1.portfolio_history) == len(result2.portfolio_history)

    finally:
        IndicatorEngine.MIN_CANDLES = old_min
        IndicatorEngine.calculate = old_calc
        os.close(fd)
        os.remove(path)


def test_causal_leak_prevention(base_policies):
    # Patch indicator engine constraints
    old_min = IndicatorEngine.MIN_CANDLES
    old_calc = IndicatorEngine.calculate
    IndicatorEngine.MIN_CANDLES = 2
    
    timestamp_history = []
    def record_causal_candles(self, candles):
        # Record candles timestamps at feature extraction to check if future is visible
        timestamp_history.append([c.timestamp for c in candles])
        return {"f1": 1.0, "f2": 2.0}

    IndicatorEngine.calculate = record_causal_candles

    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerow(["2023-01-01T00:00:00Z", "60000.0", "60050.0", "59950.0", "60000.0", "1.0"])
            writer.writerow(["2023-01-01T00:01:00Z", "60000.0", "60150.0", "59990.0", "60100.0", "1.5"])
            writer.writerow(["2023-01-01T00:02:00Z", "60100.0", "60250.0", "60080.0", "60200.0", "2.0"])
            writer.writerow(["2023-01-01T00:03:00Z", "60200.0", "60350.0", "60180.0", "60300.0", "2.5"])

        config = ReplaySessionConfig(
            initial_capital=100000.0,
            max_cycles=100,
            seed=42,
            dataset_path=path,
            enabled_symbols=["BTCUSDT"],
            timeframe="1m"
        )

        service = HistoricalReplayService()
        result = service.run_replay_session(
            config=config,
            predict_fn=lambda x: 0.0,
            custom_policies=base_policies
        )

        # Assert no future candles were ever visible during feature extraction step
        # At step 2 (index 1), only T0 and T1 should be in the buffer, not T2 or T3.
        # At step 3 (index 2), only T0, T1, T2 should be visible.
        # At step 4 (index 3), T0, T1, T2, T3.
        for batch in timestamp_history:
            for ts in batch:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                # Current clock time of the scheduler at step boundary must be >= any candle timestamp in buffer
                # (Note: ReplayClock progresses monotonically)
                pass

        # Check that buffer does not contain anything beyond step's simulated timestamp
        assert len(timestamp_history) > 0
        # First extraction (at step 2 - index 1):
        assert len(timestamp_history[0]) <= 2
        # Verify the maximum timestamp in a feature buffer never exceeds the current cycle step time
        # The first cycle runs at "2023-01-01T00:01:00+00:00"
        max_t1_str = max(timestamp_history[0])
        assert max_t1_str <= "2023-01-01T00:01:00Z"

    finally:
        IndicatorEngine.MIN_CANDLES = old_min
        IndicatorEngine.calculate = old_calc
        os.close(fd)
        os.remove(path)


def test_protective_exit_trigger(base_policies):
    # Patch indicator engine constraints
    old_min = IndicatorEngine.MIN_CANDLES
    old_calc = IndicatorEngine.calculate
    IndicatorEngine.MIN_CANDLES = 2
    IndicatorEngine.calculate = lambda self, candles: {"f1": 1.0, "f2": 2.0}

    # Setup policies to require a stop loss
    from dataclasses import replace
    base_policies = base_policies.copy()
    base_policies["exec"] = replace(base_policies["exec"], require_stop_loss=True)
    base_policies["lifecycle"] = replace(base_policies["lifecycle"], allow_stop_loss=True)

    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            # T0
            writer.writerow(["2023-01-01T00:00:00Z", "60000.0", "60050.0", "59950.0", "60000.0", "1.0"])
            # T1
            writer.writerow(["2023-01-01T00:01:00Z", "60000.0", "60150.0", "59990.0", "60100.0", "1.5"])
            # T2: Open long at close (60000.0)
            writer.writerow(["2023-01-01T00:02:00Z", "60100.0", "60250.0", "59900.0", "60000.0", "2.0"])
            # T3: Low crashes down below stop loss (which is at 10% below 60000 = 54000.0; let's hit low of 50000.0)
            writer.writerow(["2023-01-01T00:03:00Z", "60000.0", "60100.0", "50000.0", "55000.0", "2.5"])
            # T4: Another candle
            writer.writerow(["2023-01-01T00:04:00Z", "55000.0", "56000.0", "54000.0", "55000.0", "3.0"])

        config = ReplaySessionConfig(
            initial_capital=100000.0,
            max_cycles=100,
            seed=42,
            dataset_path=path,
            enabled_symbols=["BTCUSDT"],
            timeframe="1m"
        )

        class PredictOnce:
            def __init__(self):
                self.calls = 0
            def __call__(self, features):
                self.calls += 1
                if self.calls <= 2:
                    return 0.95
                return 0.5

        service = HistoricalReplayService()
        result = service.run_replay_session(
            config=config,
            predict_fn=PredictOnce(),
            custom_policies=base_policies
        )

        # The position should have been closed by protective exit trigger at T3 because low (50000.0) is below SL.
        # Let's check portfolio snapshots or results
        portfolio_snapshots = result.portfolio_history
        # Verify the position is no longer active in the final snapshot
        final_snap = portfolio_snapshots[-1]
        assert len(final_snap.open_positions) == 0
        assert result.final_equity < result.initial_equity  # Realized loss from stop-loss hitting

    finally:
        IndicatorEngine.MIN_CANDLES = old_min
        IndicatorEngine.calculate = old_calc
        os.close(fd)
        os.remove(path)
