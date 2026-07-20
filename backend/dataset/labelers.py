from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any


class BaseLabelStrategy(ABC):
    """Abstract base for all labeling strategies."""

    @abstractmethod
    def label(
        self,
        candles: Sequence,
        index: int,
        params: Dict[str, Any],
    ) -> float | None:
        """Return a label for the candle at `index`, or None if unlabelable."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FutureReturnClassification(BaseLabelStrategy):
    """
    Classifies N-bar future return:
      1  if return > +threshold
     -1  if return < -threshold
      0  otherwise (neutral)
    """

    @property
    def name(self) -> str:
        return "future_return_classification"

    def label(
        self,
        candles: Sequence,
        index: int,
        params: Dict[str, Any],
    ) -> float | None:
        horizon = params.get("horizon", 5)
        threshold = params.get("threshold", 0.01)

        future_index = index + horizon
        if future_index >= len(candles):
            return None

        current_close = float(candles[index].close)
        future_close = float(candles[future_index].close)

        if current_close == 0:
            return None

        ret = (future_close - current_close) / current_close

        if ret > threshold:
            return 1.0
        elif ret < -threshold:
            return -1.0
        else:
            return 0.0


class FutureReturnRegression(BaseLabelStrategy):
    """Returns the raw N-bar future percentage return."""

    @property
    def name(self) -> str:
        return "future_return_regression"

    def label(
        self,
        candles: Sequence,
        index: int,
        params: Dict[str, Any],
    ) -> float | None:
        horizon = params.get("horizon", 5)

        future_index = index + horizon
        if future_index >= len(candles):
            return None

        current_close = float(candles[index].close)
        future_close = float(candles[future_index].close)

        if current_close == 0:
            return None

        return (future_close - current_close) / current_close


class TPSLOutcomeLabeler(BaseLabelStrategy):
    """
    Simulates a Take-Profit / Stop-Loss bracket forward from entry candle.
    Returns 1.0 if TP is hit first, 0.0 if SL is hit first or neither within horizon.
    """

    @property
    def name(self) -> str:
        return "tpsl_outcome"

    def label(
        self,
        candles: Sequence,
        index: int,
        params: Dict[str, Any],
    ) -> float | None:
        tp_pct = params.get("tp_pct", 0.02)
        sl_pct = params.get("sl_pct", 0.01)
        horizon = params.get("horizon", 10)

        entry_close = float(candles[index].close)
        if entry_close == 0:
            return None

        tp_price = entry_close * (1 + tp_pct)
        sl_price = entry_close * (1 - sl_pct)

        for i in range(index + 1, min(index + 1 + horizon, len(candles))):
            high = float(candles[i].high)
            low = float(candles[i].low)

            if high >= tp_price:
                return 1.0
            if low <= sl_price:
                return 0.0

        return 0.0
