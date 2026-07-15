from ta.trend import ADXIndicator
import pandas as pd


class ADX:

    def calculate(self, highs, lows, closes):

        adx = ADXIndicator(
            high=pd.Series(highs),
            low=pd.Series(lows),
            close=pd.Series(closes)
        )

        return float(adx.adx().iloc[-1])