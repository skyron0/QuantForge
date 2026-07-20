import pytest
from datetime import datetime
from backend.database.models.trade import Trade
from backend.backtest.metrics import MetricsEngine


def create_mock_trade(status: str, pnl: float, open_time: datetime, close_time: datetime) -> Trade:
    return Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
        entry_price=40000.0,
        exit_price=40000.0 + pnl,
        pnl=pnl,
        status=status,
        open_time=open_time,
        close_time=close_time,
    )


def test_zero_trades():
    engine = MetricsEngine(initial_balance=10000.0)
    result = engine.calculate([])

    assert result.total_trades == 0
    assert result.winning_trades == 0
    assert result.losing_trades == 0
    assert result.win_rate == 0.0
    assert result.net_profit == 0.0
    assert result.profit_factor == 1.0
    assert len(result.equity_curve) == 0


def test_single_trade():
    engine = MetricsEngine(initial_balance=10000.0)
    t = create_mock_trade("CLOSED", 150.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 10, 30))
    result = engine.calculate([t])

    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    assert result.win_rate == 100.0
    assert result.net_profit == 150.0
    assert result.profit_factor == 150.0
    assert result.average_trade_duration == 1800.0  # 30 minutes in seconds
    assert result.maximum_drawdown_pct == 0.0
    assert result.maximum_drawdown_abs == 0.0
    assert len(result.equity_curve) == 2
    assert result.equity_curve[0].equity == 10000.0
    assert result.equity_curve[1].equity == 10150.0


def test_only_winning_trades():
    engine = MetricsEngine(initial_balance=10000.0)
    t1 = create_mock_trade("CLOSED", 100.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    t2 = create_mock_trade("CLOSED", 200.0, datetime(2026, 7, 20, 11, 0), datetime(2026, 7, 20, 12, 0))
    result = engine.calculate([t1, t2])

    assert result.total_trades == 2
    assert result.winning_trades == 2
    assert result.losing_trades == 0
    assert result.win_rate == 100.0
    assert result.gross_profit == 300.0
    assert result.gross_loss == 0.0
    assert result.profit_factor == 300.0
    assert result.maximum_drawdown_pct == 0.0


def test_only_losing_trades():
    engine = MetricsEngine(initial_balance=10000.0)
    t1 = create_mock_trade("CLOSED", -100.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    t2 = create_mock_trade("CLOSED", -200.0, datetime(2026, 7, 20, 11, 0), datetime(2026, 7, 20, 12, 0))
    result = engine.calculate([t1, t2])

    assert result.total_trades == 2
    assert result.winning_trades == 0
    assert result.losing_trades == 2
    assert result.win_rate == 0.0
    assert result.gross_profit == 0.0
    assert result.gross_loss == -300.0
    assert result.profit_factor == 0.0
    # Equity trace: 10000 -> 9900 -> 9700. Peak = 10000. Max drawdown abs = 300. Pct = 3.0%
    assert result.maximum_drawdown_abs == 300.0
    assert pytest.approx(result.maximum_drawdown_pct) == 3.0


def test_mixed_trades():
    engine = MetricsEngine(initial_balance=10000.0)
    t1 = create_mock_trade("CLOSED", 500.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    t2 = create_mock_trade("CLOSED", -300.0, datetime(2026, 7, 20, 11, 0), datetime(2026, 7, 20, 12, 0))
    t3 = create_mock_trade("CLOSED", 200.0, datetime(2026, 7, 20, 12, 0), datetime(2026, 7, 20, 13, 0))
    result = engine.calculate([t1, t2, t3])

    assert result.total_trades == 3
    assert result.winning_trades == 2
    assert result.losing_trades == 1
    assert pytest.approx(result.win_rate) == 66.666666
    assert result.net_profit == 400.0
    assert result.gross_profit == 700.0
    assert result.gross_loss == -300.0
    assert pytest.approx(result.profit_factor) == 700.0 / 300.0
    # Equity trace: 10000 -> 10500 (peak=10500) -> 10200 (peak=10500, dd=300) -> 10400 (peak=10500, dd=100)
    assert result.maximum_drawdown_abs == 300.0
    assert pytest.approx(result.maximum_drawdown_pct) == (300.0 / 10500.0 * 100.0)


def test_open_trades_ignored():
    engine = MetricsEngine(initial_balance=10000.0)
    t_closed = create_mock_trade("CLOSED", 150.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    t_open = create_mock_trade("OPEN", 500.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 12, 0))

    result = engine.calculate([t_closed, t_open])
    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.net_profit == 150.0


def test_identical_close_time_values():
    engine = MetricsEngine(initial_balance=10000.0)
    dt_close = datetime(2026, 7, 20, 11, 0)
    t1 = create_mock_trade("CLOSED", 100.0, datetime(2026, 7, 20, 10, 0), dt_close)
    t2 = create_mock_trade("CLOSED", -50.0, datetime(2026, 7, 20, 10, 30), dt_close)

    result = engine.calculate([t1, t2])
    assert result.total_trades == 2
    assert result.net_profit == 50.0


def test_floating_point_precision_cases():
    engine = MetricsEngine(initial_balance=1.0)
    # Tiny changes
    t1 = create_mock_trade("CLOSED", 0.00000001, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    t2 = create_mock_trade("CLOSED", -0.00000002, datetime(2026, 7, 20, 11, 0), datetime(2026, 7, 20, 12, 0))

    result = engine.calculate([t1, t2])
    assert pytest.approx(result.net_profit) == -0.00000001
    assert pytest.approx(result.profit_factor) == 0.00000001 / 0.00000002


def test_very_large_profit_loss_values():
    engine = MetricsEngine(initial_balance=1e12)
    t1 = create_mock_trade("CLOSED", 1e11, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    t2 = create_mock_trade("CLOSED", -2e11, datetime(2026, 7, 20, 11, 0), datetime(2026, 7, 20, 12, 0))

    result = engine.calculate([t1, t2])
    assert result.net_profit == -1e11
    assert pytest.approx(result.profit_factor) == 0.5


def test_drawdown_negative_zero_peak_edge_cases():
    # Initial balance is 0
    engine1 = MetricsEngine(initial_balance=0.0)
    # Loss trade makes equity go from 0 -> -100
    t1 = create_mock_trade("CLOSED", -100.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    result1 = engine1.calculate([t1])

    assert result1.maximum_drawdown_abs == 100.0
    assert result1.maximum_drawdown_pct == 0.0  # Guarded from division by zero since peak <= 0.0

    # Initial balance is negative
    engine2 = MetricsEngine(initial_balance=-50.0)
    t2 = create_mock_trade("CLOSED", -50.0, datetime(2026, 7, 20, 10, 0), datetime(2026, 7, 20, 11, 0))
    result2 = engine2.calculate([t2])

    assert result2.maximum_drawdown_abs == 50.0
    assert result2.maximum_drawdown_pct == 0.0  # Guarded from division by zero since peak <= 0.0
