from backend.market.consumer import MarketConsumer


class BacktestEngine:

    def __init__(self):

        self.consumer = MarketConsumer()

    def run(
        self,
        candles,
    ):

        for candle in candles:

            self.consumer.process_candle(
                candle
            )

    def close(self):

        self.consumer.close()