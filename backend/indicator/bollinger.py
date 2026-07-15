from ta.volatility import BollingerBands
import pandas as pd


class BollingerIndicator:

    def calculate(self, closes):

        closes = pd.Series(closes)

        bb = BollingerBands(close=closes)

        return {
            "upper": float(bb.bollinger_hband().iloc[-1]),
            "middle": float(bb.bollinger_mavg().iloc[-1]),
            "lower": float(bb.bollinger_lband().iloc[-1]),
        }