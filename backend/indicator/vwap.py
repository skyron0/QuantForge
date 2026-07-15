from ta.volume import VolumeWeightedAveragePrice
import pandas as pd


class VWAPIndicator:

    def calculate(self, highs, lows, closes, volumes):

        vwap = VolumeWeightedAveragePrice(
            high=pd.Series(highs),
            low=pd.Series(lows),
            close=pd.Series(closes),
            volume=pd.Series(volumes),
        )

        return float(vwap.volume_weighted_average_price().iloc[-1])