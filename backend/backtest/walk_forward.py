import datetime
from typing import List
from typing import Sequence
from datetime import timedelta
from sqlalchemy.orm import Session
from backend.database.session import engine
from backend.backtest.engine import BacktestEngine
from backend.backtest.models import BacktestResult
from backend.clock.clock import Clock
from backend.models.candle import Candle


class WalkForwardWindow:

    def __init__(
        self,
        window_id: int,
        train_start: datetime.datetime,
        train_end: datetime.datetime,
        test_start: datetime.datetime,
        test_end: datetime.datetime,
    ):
        self.window_id = window_id
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.train_result: BacktestResult | None = None
        self.test_result: BacktestResult | None = None

    def to_dict(self):
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_result": self.train_result,
            "test_result": self.test_result,
        }


class WalkForwardResult:

    def __init__(self, windows: List[WalkForwardWindow]):
        self.windows = windows
        self.global_stats = self._calculate_global_stats()

    def _calculate_global_stats(self):
        test_results = [
            w.test_result for w in self.windows if w.test_result is not None
        ]
        if not test_results:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 1.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "maximum_drawdown_pct": 0.0,
                "maximum_drawdown_abs": 0.0,
            }

        total_trades = sum(r.total_trades for r in test_results)
        winning_trades = sum(r.winning_trades for r in test_results)
        losing_trades = sum(r.losing_trades for r in test_results)

        net_profit = sum(r.net_profit for r in test_results)
        gross_profit = sum(r.gross_profit for r in test_results)
        gross_loss = sum(r.gross_loss for r in test_results)

        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        if gross_loss != 0.0:
            profit_factor = gross_profit / abs(gross_loss)
        else:
            profit_factor = gross_profit if gross_profit > 0 else 1.0

        largest_win = max((r.largest_win for r in test_results), default=0.0)
        largest_loss = min((r.largest_loss for r in test_results), default=0.0)

        maximum_drawdown_pct = max(
            (r.maximum_drawdown_pct for r in test_results), default=0.0
        )
        maximum_drawdown_abs = max(
            (r.maximum_drawdown_abs for r in test_results), default=0.0
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "maximum_drawdown_pct": maximum_drawdown_pct,
            "maximum_drawdown_abs": maximum_drawdown_abs,
        }


class WalkForwardEngine:

    def __init__(self, db_engine=None):
        self.db_engine = db_engine if db_engine is not None else engine

    def generate_windows(
        self,
        candles: Sequence[Candle],
        train_days: int,
        test_days: int,
        step_days: int,
    ) -> List[WalkForwardWindow]:
        if not candles:
            return []

        sorted_candles = sorted(candles, key=lambda c: c.open_time)
        start_time = sorted_candles[0].open_time
        end_time = sorted_candles[-1].open_time

        windows = []
        window_id = 1
        current_train_start = start_time

        while True:
            current_train_end = current_train_start + timedelta(days=train_days)
            current_test_start = current_train_end
            current_test_end = current_test_start + timedelta(days=test_days)

            if current_test_start >= end_time:
                break

            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=current_train_end,
                    test_start=current_test_start,
                    test_end=current_test_end,
                )
            )

            window_id += 1
            current_train_start = current_train_start + timedelta(days=step_days)

        return windows

    def run(
        self,
        candles: Sequence[Candle],
        train_days: int,
        test_days: int,
        step_days: int,
        strategy_name: str = "rule_based",
        strategy_params: dict | None = None,
    ) -> WalkForwardResult:
        windows = self.generate_windows(candles, train_days, test_days, step_days)

        for w in windows:
            train_candles = [
                c for c in candles if w.train_start <= c.open_time <= w.train_end
            ]
            test_candles = [
                c for c in candles if w.test_start <= c.open_time <= w.test_end
            ]

            if train_candles:
                w.train_result = self._run_single_backtest(
                    train_candles, strategy_name, strategy_params
                )
            else:
                w.train_result = self._empty_result()

            if test_candles:
                w.test_result = self._run_single_backtest(
                    test_candles, strategy_name, strategy_params
                )
            else:
                w.test_result = self._empty_result()

        return WalkForwardResult(windows)

    def _run_single_backtest(
        self,
        candles: Sequence[Candle],
        strategy_name: str,
        strategy_params: dict | None,
    ) -> BacktestResult:
        connection = self.db_engine.connect()
        transaction = connection.begin()
        db = Session(bind=connection)

        try:
            from backend.database.base import Base

            Base.metadata.create_all(bind=connection)

            from backend.strategy.registry import StrategyLoader

            params = strategy_params or {}
            strategy = StrategyLoader.load_strategy(strategy_name, **params)

            clock = Clock()
            backtest_engine = BacktestEngine(
                db_session=db, clock=clock, strategy=strategy
            )
            backtest_engine.run(candles)

            result = backtest_engine.get_metrics()
            backtest_engine.close()
            return result
        finally:
            transaction.rollback()
            db.close()
            connection.close()

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            net_profit=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            average_profit=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=1.0,
            largest_win=0.0,
            largest_loss=0.0,
            average_trade_duration=0.0,
            maximum_drawdown_pct=0.0,
            maximum_drawdown_abs=0.0,
            equity_curve=[],
        )
