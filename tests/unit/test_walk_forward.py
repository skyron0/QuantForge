import datetime
import pytest
from sqlalchemy import create_engine
from backend.models.candle import Candle
from backend.backtest.models import BacktestResult
from backend.backtest.walk_forward import (
    WalkForwardEngine,
    WalkForwardWindow,
    WalkForwardResult,
)


def _create_mock_candle(
    symbol: str, open_time: datetime.datetime, close_price: float
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe="1h",
        open=close_price - 1.0,
        high=close_price + 2.0,
        low=close_price - 2.0,
        close=close_price,
        volume=100.0,
        open_time=open_time,
    )


def test_walk_forward_split_generation():
    """
    Verify WalkForwardEngine generates correct chronological train/test windows.
    """
    engine = WalkForwardEngine()

    start_date = datetime.datetime(2026, 7, 1, 12, 0, 0)
    candles = [
        _create_mock_candle(
            "BTCUSDT", start_date + datetime.timedelta(days=i), 50000.0
        )
        for i in range(30)
    ]

    # total span of candles: 2026-07-01 to 2026-07-30
    # train_days=10, test_days=5, step_days=5
    windows = engine.generate_windows(
        candles, train_days=10, test_days=5, step_days=5
    )

    # Expected:
    # W1: Train=Jul 1 -> Jul 11, Test=Jul 11 -> Jul 16
    # W2: Train=Jul 6 -> Jul 16, Test=Jul 16 -> Jul 21
    # W3: Train=Jul 11 -> Jul 21, Test=Jul 21 -> Jul 26
    # W4: Train=Jul 16 -> Jul 26, Test=Jul 26 -> Jul 31 (start Jul 26 < Jul 30)
    assert len(windows) == 4

    w1 = windows[0]
    assert w1.window_id == 1
    assert w1.train_start == start_date
    assert w1.train_end == start_date + datetime.timedelta(days=10)
    assert w1.test_start == start_date + datetime.timedelta(days=10)
    assert w1.test_end == start_date + datetime.timedelta(days=15)

    w4 = windows[3]
    assert w4.window_id == 4
    assert w4.train_start == start_date + datetime.timedelta(days=15)
    assert w4.test_start == start_date + datetime.timedelta(days=25)


def test_walk_forward_result_aggregation():
    """
    Verify WalkForwardResult aggregates statistics correctly from individual windows.
    """
    w1 = WalkForwardWindow(
        window_id=1,
        train_start=datetime.datetime(2026, 7, 1),
        train_end=datetime.datetime(2026, 7, 11),
        test_start=datetime.datetime(2026, 7, 11),
        test_end=datetime.datetime(2026, 7, 16),
    )
    # Win rate: 100%, Net profit: 500, pf: 1.0, DD: 2%
    w1.test_result = BacktestResult(
        total_trades=2,
        winning_trades=2,
        losing_trades=0,
        win_rate=100.0,
        net_profit=500.0,
        gross_profit=500.0,
        gross_loss=0.0,
        average_profit=250.0,
        average_win=250.0,
        average_loss=0.0,
        profit_factor=1.0,
        largest_win=300.0,
        largest_loss=0.0,
        average_trade_duration=120.0,
        maximum_drawdown_pct=2.0,
        maximum_drawdown_abs=10.0,
        equity_curve=[],
    )

    w2 = WalkForwardWindow(
        window_id=2,
        train_start=datetime.datetime(2026, 7, 6),
        train_end=datetime.datetime(2026, 7, 16),
        test_start=datetime.datetime(2026, 7, 16),
        test_end=datetime.datetime(2026, 7, 21),
    )
    # Win rate: 50%, Net profit: -100, Gross profit: 100, Gross Loss: -200, DD: 4%
    w2.test_result = BacktestResult(
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
        net_profit=-100.0,
        gross_profit=100.0,
        gross_loss=-200.0,
        average_profit=-50.0,
        average_win=100.0,
        average_loss=-200.0,
        profit_factor=0.5,
        largest_win=100.0,
        largest_loss=-200.0,
        average_trade_duration=180.0,
        maximum_drawdown_pct=4.0,
        maximum_drawdown_abs=25.0,
        equity_curve=[],
    )

    result = WalkForwardResult([w1, w2])
    stats = result.global_stats

    # Aggregated assertions:
    assert stats["total_trades"] == 4
    assert stats["winning_trades"] == 3
    assert stats["losing_trades"] == 1
    assert stats["win_rate"] == 75.0  # 3/4 * 100
    assert stats["net_profit"] == 400.0  # 500 - 100
    assert stats["gross_profit"] == 600.0  # 500 + 100
    assert stats["gross_loss"] == -200.0  # 0 - 200
    assert stats["profit_factor"] == 3.0  # 600 / 200
    assert stats["largest_win"] == 300.0
    assert stats["largest_loss"] == -200.0
    assert stats["maximum_drawdown_pct"] == 4.0  # Max of 2% and 4%
    assert stats["maximum_drawdown_abs"] == 25.0  # Max of 10 and 25


def test_walk_forward_engine_run_with_memory_db():
    """
    Verify WalkForwardEngine runs successfully and transaction rollbacks prevent cross-talk.
    """
    # Create in-memory SQLite engine
    memory_db_engine = create_engine("sqlite:///:memory:")
    wfa_engine = WalkForwardEngine(db_engine=memory_db_engine)

    # Create dummy candles spanning 25 days, enough to run 2 windows
    # train=10, test=5, step=5 ->
    # Window 1: Train=0->10, Test=10->15
    # Window 2: Train=5->15, Test=15->20
    # Window 3: Train=10->20, Test=20->25
    start_date = datetime.datetime(2026, 7, 1, 0, 0, 0)
    candles = [
        _create_mock_candle(
            "BTCUSDT", start_date + datetime.timedelta(days=i), 50000.0
        )
        for i in range(25)
    ]

    result = wfa_engine.run(
        candles=candles, train_days=10, test_days=5, step_days=5
    )

    # We expect 3 windows
    assert len(result.windows) == 3
    for w in result.windows:
        assert w.train_result is not None
        assert w.test_result is not None
        # Verify that each window has its own metrics generated without errors
        assert isinstance(w.train_result, BacktestResult)
        assert isinstance(w.test_result, BacktestResult)

    # The in-memory database should be completely clean because of rollbacks
    with memory_db_engine.connect() as con:
        # Check if tables exist but contain no rows or tables were dropped/rolled back.
        # SQLite system query to confirm no data remains:
        from sqlalchemy import inspect, text

        inspector = inspect(memory_db_engine)
        if "trades" in inspector.get_table_names():
            row_count = con.execute(text("SELECT COUNT(*) FROM trades")).scalar()  # type: ignore
            assert row_count == 0
