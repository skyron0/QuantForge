from typing import Sequence
from backend.database.models.trade import Trade
from backend.backtest.models import BacktestResult, EquityPoint


class MetricsEngine:

    def __init__(self, initial_balance: float = 10000.0):

        self.initial_balance = initial_balance

    def calculate(self, trades: Sequence[Trade]) -> BacktestResult:

        closed_trades = [t for t in trades if t.status == "CLOSED"]
        closed_trades.sort(key=lambda t: t.close_time)

        total_trades = len(closed_trades)

        if total_trades == 0:

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

        winning_trades_list = [t for t in closed_trades if float(t.pnl) > 0.0]  # type: ignore

        losing_trades_list = [t for t in closed_trades if float(t.pnl) <= 0.0]  # type: ignore

        winning_trades = len(winning_trades_list)

        losing_trades = len(losing_trades_list)

        win_rate = (winning_trades / total_trades) * 100.0

        gross_profit = float(sum(t.pnl for t in winning_trades_list))  # type: ignore

        gross_loss = float(sum(t.pnl for t in losing_trades_list))  # type: ignore

        net_profit = gross_profit + gross_loss

        average_profit = net_profit / total_trades

        average_win = (
            (gross_profit / winning_trades)
            if winning_trades > 0
            else 0.0
        )

        average_loss = (
            (gross_loss / losing_trades)
            if losing_trades > 0
            else 0.0
        )

        if gross_loss == 0.0:

            profit_factor = (
                gross_profit if gross_profit > 0.0 else 1.0
            )

        else:

            profit_factor = gross_profit / abs(gross_loss)

        largest_win = float(max(
            (t.pnl for t in winning_trades_list),  # type: ignore
            default=0.0,
        ))

        largest_loss = float(min(
            (t.pnl for t in losing_trades_list),  # type: ignore
            default=0.0,
        ))

        total_duration = 0.0

        for t in closed_trades:

            if t.close_time and t.open_time:

                td = t.close_time - t.open_time  # type: ignore
                total_duration += td.total_seconds()

        average_trade_duration = (
            total_duration / total_trades
        )

        equity_curve = []

        current_equity = self.initial_balance

        if closed_trades:

            equity_curve.append(
                EquityPoint(
                    timestamp=closed_trades[0].open_time,  # type: ignore
                    equity=current_equity,
                    drawdown_pct=0.0,
                    drawdown_abs=0.0,
                )
            )

        peak = current_equity

        max_drawdown_pct = 0.0

        max_drawdown_abs = 0.0

        for t in closed_trades:

            current_equity += float(t.pnl)  # type: ignore

            if current_equity > peak:

                peak = current_equity

            dd_abs = peak - current_equity

            dd_pct = (
                (dd_abs / peak * 100.0)
                if peak > 0.0
                else 0.0
            )

            if dd_pct > max_drawdown_pct:

                max_drawdown_pct = dd_pct

            if dd_abs > max_drawdown_abs:

                max_drawdown_abs = dd_abs

            equity_curve.append(
                EquityPoint(
                    timestamp=t.close_time,  # type: ignore
                    equity=current_equity,
                    drawdown_pct=dd_pct,
                    drawdown_abs=dd_abs,
                )
            )

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            net_profit=net_profit,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            average_profit=average_profit,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=average_trade_duration,
            maximum_drawdown_pct=max_drawdown_pct,
            maximum_drawdown_abs=max_drawdown_abs,
            equity_curve=equity_curve,
        )