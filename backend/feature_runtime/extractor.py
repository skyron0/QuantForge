"""
Feature extraction using the existing IndicatorEngine.

Responsible for converting raw candle history into an ordered feature
vector.  Enforces causal constraints: only candles with timestamp <= T
are used when generating features for timestamp T.
"""

from typing import List, Optional

from backend.feature_runtime.buffer import HistoricalFeatureBuffer, BufferCandle
from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.models import FeatureSnapshot
from backend.feature_runtime.exceptions import (
    InsufficientHistoryError,
    FeatureWarmupError,
    InvalidFeatureValueError,
    FutureFeatureError,
)

import math


class _IndicatorProxy:
    """
    Thin wrapper around the existing IndicatorEngine.

    IndicatorEngine is imported lazily so that feature_runtime has no
    module-level coupling to configs.settings (which requires .env).
    """

    def __init__(self) -> None:
        self._engine: Optional[object] = None

    def _ensure_engine(self) -> object:
        if self._engine is None:
            from backend.indicator.indicator_engine import IndicatorEngine
            self._engine = IndicatorEngine()
        return self._engine

    def calculate(self, candles: list) -> Optional[dict]:
        eng = self._ensure_engine()
        return eng.calculate(candles)  # type: ignore[union-attr]


class _CandleAdapter:
    """Adapts BufferCandle to the duck-typed interface expected by IndicatorEngine."""

    def __init__(self, bc: BufferCandle) -> None:
        self.open = bc.open
        self.high = bc.high
        self.low = bc.low
        self.close = bc.close
        self.volume = bc.volume
        self.timestamp = bc.timestamp


class FeatureExtractor:
    """
    Extracts an ordered feature vector from historical candles via the
    IndicatorEngine, then returns a FeatureSnapshot matching the given
    FeatureSchema.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        buffer: HistoricalFeatureBuffer,
        minimum_history: int = 100,
    ) -> None:
        self._schema = schema
        self._buffer = buffer
        self._minimum_history = minimum_history
        self._indicator_proxy = _IndicatorProxy()

    # ── public ────────────────────────────────────────────────────────────

    def extract(self, symbol: str, timestamp: str) -> FeatureSnapshot:
        """
        Build a FeatureSnapshot for *symbol* at *timestamp*.

        Only candles with timestamp <= *timestamp* are considered (causal
        guarantee).
        """

        # 1. Causal filter
        candles = self._buffer.get_candles_up_to(symbol, timestamp)

        if len(candles) < self._minimum_history:
            raise InsufficientHistoryError(
                f"{symbol}: {len(candles)} candles available, "
                f"{self._minimum_history} required"
            )

        # 2. Delegate to IndicatorEngine
        adapted = [_CandleAdapter(c) for c in candles]
        raw = self._indicator_proxy.calculate(adapted)
        if raw is None:
            raise FeatureWarmupError(
                f"IndicatorEngine returned None for {symbol} — still warming up"
            )

        # 3. Build ordered feature vector using schema's canonical ordering
        names: List[str] = []
        values: List[float] = []
        for name in self._schema.feature_names:
            if name not in raw:
                raise InvalidFeatureValueError(
                    f"Feature '{name}' missing from indicator output"
                )
            val = float(raw[name])
            if math.isnan(val) or math.isinf(val):
                raise InvalidFeatureValueError(
                    f"Feature '{name}' has invalid value: {val}"
                )
            names.append(name)
            values.append(val)

        return FeatureSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            feature_names=names,
            feature_values=values,
            feature_version=self._schema.schema_version,
            schema_fingerprint=self._schema.fingerprint,
        )
