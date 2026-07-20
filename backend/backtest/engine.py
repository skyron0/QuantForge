from backend.market.consumer import MarketConsumer
from backend.backtest.metrics import MetricsEngine


class BacktestEngine:

    def __init__(self, db_session=None, clock=None, strategy=None):

        self.db = db_session
        self.clock = clock
        self.consumer = MarketConsumer(db_session=db_session, clock=clock, strategy=strategy)

    def run(
        self,
        candles,
    ):

        for candle in candles:

            if self.clock is not None:
                self.clock.set_time(candle.open_time)

            self.consumer.feed_candle(
                candle
            )

    def get_metrics(self):

        if self.db is None:
            return MetricsEngine(initial_balance=10000.0).calculate([])
        from backend.database.models.trade import Trade
        trades = self.db.query(Trade).all()
        metrics = MetricsEngine(initial_balance=10000.0)
        return metrics.calculate(trades)

    def close(self):

        self.consumer.close()