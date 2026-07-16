import math

import pandas as pd
from ta.volatility import BollingerBands


class BollingerIndicator:

    def calculate(
        self,
        closes,
        window: int = 20,
        std: int = 2,
    ):

        if len(closes) < window:
            return None

        series = pd.Series(closes, dtype="float64")

        bb = BollingerBands(
            close=series,
            window=window,
            window_dev=std,
        )

        upper = bb.bollinger_hband().iloc[-1]
        middle = bb.bollinger_mavg().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]

        if (
            pd.isna(upper)
            or pd.isna(middle)
            or pd.isna(lower)
        ):
            return None

        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
        }