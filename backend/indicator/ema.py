import pandas as pd
import ta


class EMAIndicator:

    def calculate(self, closes, period=20):

        series = pd.Series(closes)

        return ta.trend.EMAIndicator(
            close=series,
            window=period
        ).ema_indicator().iloc[-1]