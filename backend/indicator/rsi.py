import math

import pandas as pd
import ta


class RSIIndicator:

    def calculate(
        self,
        closes,
        period: int = 14,
    ):

        if len(closes) < period + 1:
            return None

        series = pd.Series(closes, dtype="float64")

        value = ta.momentum.RSIIndicator(
            close=series,
            window=period
        ).rsi().iloc[-1]

        if pd.isna(value) or math.isnan(float(value)):
            return None

        return float(value)