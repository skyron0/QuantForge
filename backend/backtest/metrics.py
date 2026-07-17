from backend.database.session import SessionLocal
from backend.database.models.trade import Trade


class Metrics:

    def __init__(self):

        self.db = SessionLocal()

    def total_trades(self):

        return self.db.query(Trade).count()

    def closed_trades(self):

        return (
            self.db.query(Trade)
            .filter(
                Trade.status == "CLOSED"
            )
            .all()
        )

    def win_rate(self):

        trades = self.closed_trades()

        if not trades:
            return 0.0

        wins = len(
            [
                t
                for t in trades
                if t.pnl > 0
            ]
        )

        return wins / len(trades) * 100

    def net_profit(self):

        trades = self.closed_trades()

        return sum(
            t.pnl
            for t in trades
        )

    def average_profit(self):

        trades = self.closed_trades()

        if not trades:
            return 0

        return sum(
            t.pnl
            for t in trades
        ) / len(trades)