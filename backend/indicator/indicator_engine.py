from backend.indicator.rsi import RSIIndicator
from backend.indicator.ema import EMAIndicator


class IndicatorEngine:

    def __init__(self):

        self.rsi = RSIIndicator()
        self.ema = EMAIndicator()

    def calculate(self, candles):

        closes = [c.close for c in candles]

        if len(closes) < 20:
            return None

        return {
            "rsi": self.rsi.calculate(closes),
            "ema20": self.ema.calculate(closes, 20)
        }