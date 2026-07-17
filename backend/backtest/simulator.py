from backend.backtest.engine import BacktestEngine


class Simulator:

    def __init__(self):

        self.engine = BacktestEngine()

    def simulate(
        self,
        candles,
    ):

        self.engine.run(candles)

        self.engine.close()