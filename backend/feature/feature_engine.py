from backend.feature.models import FeatureSet

from backend.indicator.rsi import RSIIndicator
from backend.indicator.ema import EMAIndicator
from backend.indicator.macd import MACDIndicator
from backend.indicator.bollinger import BollingerIndicator
from backend.indicator.atr import ATRIndicator
from backend.indicator.adx import ADX
from backend.indicator.vwap import VWAPIndicator


class FeatureEngine:

    def __init__(self):

        self.rsi = RSIIndicator()
        self.ema = EMAIndicator()
        self.macd = MACDIndicator()
        self.bb = BollingerIndicator()
        self.atr = ATRIndicator()
        self.adx = ADX()
        self.vwap = VWAPIndicator()

    def build(self, candles):

        if len(candles) < 30:
            return None

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        macd = self.macd.calculate(closes)
        bb = self.bb.calculate(closes)

        return FeatureSet(

            symbol=candles[-1].symbol,
            
            close=candles[-1].close,

            rsi=self.rsi.calculate(closes),

            ema20=self.ema.calculate(closes, 20),

            macd=macd["macd"],
            macd_signal=macd["signal"],
            macd_histogram=macd["histogram"],

            bollinger_upper=bb["upper"],
            bollinger_middle=bb["middle"],
            bollinger_lower=bb["lower"],

            atr=self.atr.calculate(highs, lows, closes),

            adx=self.adx.calculate(highs, lows, closes),

            vwap=self.vwap.calculate(
                highs,
                lows,
                closes,
                volumes
            )
        )