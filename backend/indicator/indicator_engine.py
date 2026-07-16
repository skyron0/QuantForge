from backend.indicator.rsi import RSIIndicator
from backend.indicator.ema import EMAIndicator
from backend.indicator.macd import MACDIndicator
from backend.indicator.bollinger import BollingerIndicator
from backend.indicator.atr import ATRIndicator
from backend.indicator.adx import ADX
from backend.indicator.vwap import VWAPIndicator


class IndicatorEngine:

    def __init__(self):

        self.rsi = RSIIndicator()
        self.ema = EMAIndicator()
        self.macd = MACDIndicator()
        self.bollinger = BollingerIndicator()
        self.atr = ATRIndicator()
        self.adx = ADX()
        self.vwap = VWAPIndicator()

    def calculate(self, candles):

        if len(candles) < 35:
            return None

        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        volumes = [float(c.volume) for c in candles]

        rsi = self.rsi.calculate(closes)
        ema20 = self.ema.calculate(closes, 20)
        macd = self.macd.calculate(closes)
        atr = self.atr.calculate(highs, lows, closes)
        adx = self.adx.calculate(highs, lows, closes)
        vwap = self.vwap.calculate(
            highs,
            lows,
            closes,
            volumes,
        )
        bollinger = self.bollinger.calculate(closes)

        if (
            rsi is None
            or ema20 is None
            or macd is None
            or atr is None
            or adx is None
            or vwap is None
            or bollinger is None
        ):
            return None

        return {
            "rsi": rsi,
            "ema20": ema20,

            "macd": macd["macd"],
            "macd_signal": macd["signal"],
            "macd_histogram": macd["histogram"],

            "atr": atr,
            "adx": adx,
            "vwap": vwap,

            "bb_upper": bollinger["upper"],
            "bb_middle": bollinger["middle"],
            "bb_lower": bollinger["lower"],
        }