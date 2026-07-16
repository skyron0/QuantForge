from configs.logging import app_logger


class TradeHistory:

    def __init__(self):

        self.trades = []

    def add_trade(self, trade):

        self.trades.append(trade)

        app_logger.info(

            f"[TRADE CLOSED] "
            f"{trade.symbol} | "
            f"{trade.side} | "
            f"PnL={trade.pnl:.2f} | "
            f"Reason={trade.reason}"

        )

    def total_trades(self):

        return len(self.trades)

    def wins(self):

        return len(

            [t for t in self.trades if t.pnl > 0]

        )

    def losses(self):

        return len(

            [t for t in self.trades if t.pnl <= 0]

        )

    def total_profit(self):

        return sum(

            t.pnl

            for t in self.trades

        )