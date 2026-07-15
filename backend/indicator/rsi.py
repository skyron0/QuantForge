import pandas as pd
import ta


class RSIIndicator:

    def calculate(self, closes, period=14):

        series = pd.Series(closes)

        return ta.momentum.RSIIndicator(
            close=series,
            window=period
        ).rsi().iloc[-1]