from configs.logging import app_logger

from backend.decision.models import Decision


class DecisionEngine:

    BUY_THRESHOLD = 40
    SELL_THRESHOLD = -40

    def decide(self, features):

        if features is None:
            return None

        score = 0
        reasons = []

        # -------------------------
        # RSI
        # -------------------------
        if features.rsi <= 30:
            score += 30
            reasons.append("RSI Oversold")

        elif features.rsi >= 70:
            score -= 30
            reasons.append("RSI Overbought")

        # -------------------------
        # MACD Histogram
        # -------------------------
        if features.macd_histogram > 0:
            score += 20
            reasons.append("MACD Bullish")

        elif features.macd_histogram < 0:
            score -= 20
            reasons.append("MACD Bearish")

        # -------------------------
        # ADX
        # -------------------------
        if features.adx >= 25:
            score += 15
            reasons.append("Strong Trend")

        # -------------------------
        # EMA Trend
        # -------------------------
        if features.close > features.ema20:
            score += 10
            reasons.append("Above EMA20")

        else:
            score -= 10
            reasons.append("Below EMA20")

        # -------------------------
        # VWAP
        # -------------------------
        if features.close > features.vwap:
            score += 10
            reasons.append("Above VWAP")

        else:
            score -= 10
            reasons.append("Below VWAP")

        # -------------------------
        # Bollinger Bands
        # -------------------------
        if features.close <= features.bb_lower:
            score += 15
            reasons.append("Lower BB")

        elif features.close >= features.bb_upper:
            score -= 15
            reasons.append("Upper BB")

        # -------------------------
        # Confidence
        # -------------------------
        confidence = min(abs(score) / 100.0, 0.99)

        app_logger.info(
            f"[DECISION] Score={score} "
            f"Confidence={confidence:.2f} "
            f"Reasons={', '.join(reasons)}"
        )

        # -------------------------
        # BUY
        # -------------------------
        if score >= self.BUY_THRESHOLD:

            return Decision(
                action="BUY",
                confidence=confidence,
                reason=", ".join(reasons)
            )

        # -------------------------
        # SELL
        # -------------------------
        if score <= self.SELL_THRESHOLD:

            return Decision(
                action="SELL",
                confidence=confidence,
                reason=", ".join(reasons)
            )

        # -------------------------
        # HOLD
        # -------------------------
        return Decision(
            action="HOLD",
            confidence=max(confidence, 0.50),
            reason=", ".join(reasons) if reasons else "Neutral"
        )