from backend.decision.models import Decision


class DecisionEngine:

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
                confidence=0.80,
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
                confidence=0.80,
                reason="RSI Overbought + EMA Trend + MACD Bearish"
            )

        return Decision(
            action="HOLD",
            confidence=0.50,
            reason="No confirmation"
        )