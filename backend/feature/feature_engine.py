from backend.feature.models import FeatureVector


class FeatureEngine:

    def build(
        self,
        candle,
        indicators,
    ):

        return FeatureVector(

            symbol=candle.symbol,

            rsi=indicators["rsi"],

            ema20=indicators["ema20"],

            macd=indicators["macd"],

            macd_signal=indicators["macd_signal"],

            macd_histogram=indicators["macd_histogram"],

            adx=indicators["adx"],

            atr=indicators["atr"],

            bb_upper=indicators["bb_upper"],

            bb_middle=indicators["bb_middle"],

            bb_lower=indicators["bb_lower"],

            vwap=indicators["vwap"],

            close=candle.close,
        )