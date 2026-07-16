import math

import pandas as pd
from ta.trend import MACD


class MACDIndicator:

    def calculate(self, closes):

        if len(closes) < 35:
            return None

        series = pd.Series(closes, dtype="float64")

        macd = MACD(close=series)

        macd_value = macd.macd().iloc[-1]
        signal_value = macd.macd_signal().iloc[-1]
        histogram_value = macd.macd_diff().iloc[-1]

        if (
            pd.isna(macd_value)
            or pd.isna(signal_value)
            or pd.isna(histogram_value)
        ):
            return None

        return {
            "macd": float(macd_value),
            "signal": float(signal_value),
            "histogram": float(histogram_value),
        }