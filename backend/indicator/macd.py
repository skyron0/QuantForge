from ta.trend import MACD
import pandas as pd


class MACDIndicator:

    def calculate(self, closes):

        closes = pd.Series(closes)

        macd = MACD(close=closes)

        return {
            "macd": float(macd.macd().iloc[-1]),
            "signal": float(macd.macd_signal().iloc[-1]),
            "histogram": float(macd.macd_diff().iloc[-1]),
        }