from backend.decision.models import Decision


class DecisionEngine:

    BUY_CONFIDENCE = 0.80
    SELL_CONFIDENCE = 0.80
    HOLD_CONFIDENCE = 0.50

    def decide(self, features):

        if features is None:
            return None

        # BUY
        if (
            features.rsi < 30
            and features.close > features.ema20
            and features.macd > features.macd_signal
        ):

            return Decision(
                action="BUY",
                confidence=self.BUY_CONFIDENCE,
                reason="RSI Oversold + EMA Trend + MACD Bullish"
            )

        # SELL
        if (
            features.rsi > 70
            and features.close < features.ema20
            and features.macd < features.macd_signal
        ):

            return Decision(
                action="SELL",
                confidence=self.SELL_CONFIDENCE,
                reason="RSI Overbought + EMA Trend + MACD Bearish"
            )

        return Decision(
            action="HOLD",
            confidence=self.HOLD_CONFIDENCE,
            reason="No confirmation"
        )