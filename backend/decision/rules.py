from backend.decision.models import Decision
from backend.decision.signals import BUY, SELL, HOLD


class RuleEngine:

    def evaluate(self, feature):

        if feature.rsi < 30 and feature.close > feature.ema20:
            return Decision(
                action=BUY,
                confidence=0.80,
                reason="Oversold + EMA confirmation"
            )

        if feature.rsi > 70 and feature.close < feature.ema20:
            return Decision(
                action=SELL,
                confidence=0.80,
                reason="Overbought + EMA confirmation"
            )

        return Decision(
            action=HOLD,
            confidence=0.50,
            reason="No signal"
        )