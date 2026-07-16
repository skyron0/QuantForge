import math

import pandas as pd
from ta.volume import VolumeWeightedAveragePrice


class VWAPIndicator:

    def calculate(
        self,
        highs,
        lows,
        closes,
        volumes,
    ):

        if (
            len(highs) < 2
            or len(lows) < 2
            or len(closes) < 2
            or len(volumes) < 2
        ):
            return None

        vwap = VolumeWeightedAveragePrice(
            high=pd.Series(highs, dtype="float64"),
            low=pd.Series(lows, dtype="float64"),
            close=pd.Series(closes, dtype="float64"),
            volume=pd.Series(volumes, dtype="float64"),
        )

        value = vwap.volume_weighted_average_price().iloc[-1]

        if pd.isna(value) or math.isnan(float(value)):
            return None

        return float(value)