from ta.volatility import AverageTrueRange
import pandas as pd


class ATRIndicator:

    def calculate(self, highs, lows, closes):

        atr = AverageTrueRange(
            high=pd.Series(highs),
            low=pd.Series(lows),
            close=pd.Series(closes)
        )

        return float(atr.average_true_range().iloc[-1])