from backend.backtest.engine import BacktestEngine


class Simulator:

    def __init__(self, db_session=None, clock=None):

        self.engine = BacktestEngine(db_session=db_session, clock=clock)

    def simulate(
        self,
        candles,
    ):

        self.engine.run(candles)

        self.engine.close()