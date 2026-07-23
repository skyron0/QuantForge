"""
Signal generation: maps PredictionResult to an MLSignal.

Uses configurable thresholds to determine BULLISH / BEARISH / NEUTRAL
direction from the scalar prediction output.
"""

import uuid
from datetime import datetime, timezone

from backend.feature_runtime.models import PredictionResult
from backend.feature_runtime.exceptions import SignalGenerationError
from backend.decision.models import MLSignal


class FeatureSignalMapper:
    """
    Stateless mapper converting a PredictionResult into a directional
    MLSignal.

    The prediction is assumed to be a probability-like value in [0, 1]:
      - prediction >= bullish_threshold  →  BULLISH
      - prediction <= bearish_threshold  →  BEARISH
      - otherwise                        →  NEUTRAL
    """

    def __init__(
        self,
        bullish_threshold: float = 0.55,
        bearish_threshold: float = 0.45,
        default_timeframe: str = "5m",
    ) -> None:
        self._bullish = bullish_threshold
        self._bearish = bearish_threshold
        self._timeframe = default_timeframe

    def map(self, result: PredictionResult) -> MLSignal:
        """Convert *result* into an MLSignal."""
        try:
            direction = self._resolve_direction(result.prediction)
            confidence = self._resolve_confidence(result.prediction)

            return MLSignal(
                model_version=result.model_version or "feature-runtime",
                symbol=result.symbol,
                timeframe=self._timeframe,
                prediction=result.prediction,
                direction=direction,
                confidence=confidence,
                calibrated=result.is_calibrated,
                timestamp=result.timestamp,
                drift_status="normal",
                metadata={
                    "feature_version": result.feature_version,
                    "latency_ms": result.latency_ms,
                },
            )
        except Exception as exc:
            raise SignalGenerationError(
                f"Failed to map prediction to signal: {exc}"
            ) from exc

    # ── internal ──────────────────────────────────────────────────────────

    def _resolve_direction(self, prediction: float) -> str:
        if prediction >= self._bullish:
            return "BULLISH"
        if prediction <= self._bearish:
            return "BEARISH"
        return "NEUTRAL"

    def _resolve_confidence(self, prediction: float) -> float:
        """
        Confidence = distance from the 0.5 neutral midpoint, scaled to [0, 1].
        """
        return min(abs(prediction - 0.5) * 2.0, 1.0)
