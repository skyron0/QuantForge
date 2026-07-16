import math

import pandas as pd
import ta


class EMAIndicator:

    def calculate(
        self,
        closes,
        period: int = 20,
    ):

        if len(closes) < period:
            return None

        series = pd.Series(closes, dtype="float64")

        value = ta.trend.EMAIndicator(
            close=series,
            window=period
        ).ema_indicator().iloc[-1]

        if pd.isna(value) or math.isnan(float(value)):
            return None

        return float(value)