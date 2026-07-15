from collections import defaultdict

from backend.market.models import MarketTick


class CandleAggregator:

    def __init__(self):

        self.builders = {}

    def process_tick(self, tick: MarketTick):

        symbol = tick.symbol

        if symbol not in self.builders:

            from backend.market.candle.builder import CandleBuilder

            self.builders[symbol] = CandleBuilder(symbol)

        return self.builders[symbol].update(tick)