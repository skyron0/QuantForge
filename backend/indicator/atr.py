import math

import pandas as pd
from ta.volatility import AverageTrueRange


class ATRIndicator:

    def calculate(
        self,
        highs,
        lows,
        closes,
        period: int = 14,
    ):

        if (
            len(highs) < period
            or len(lows) < period
            or len(closes) < period
        ):
            return None

        atr = AverageTrueRange(
            high=pd.Series(highs, dtype="float64"),
            low=pd.Series(lows, dtype="float64"),
            close=pd.Series(closes, dtype="float64"),
            window=period,
        )

        value = atr.average_true_range().iloc[-1]

        if pd.isna(value) or math.isnan(float(value)):
            return None

        return float(value)