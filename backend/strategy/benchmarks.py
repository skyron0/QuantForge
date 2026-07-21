import numpy as np
from backend.strategy.base import BaseStrategy
from backend.strategy.registry import StrategyRegistry
from backend.decision.models import Decision


@StrategyRegistry.register("buy_and_hold")
class BuyAndHoldStrategy(BaseStrategy):
    """Always generates BUY decisions representing Buy & Hold benchmark."""
    def decide(self, features) -> Decision | None:
        return Decision(action="BUY", confidence=1.0, reason="Buy & Hold benchmark")


@StrategyRegistry.register("always_flat")
class AlwaysFlatStrategy(BaseStrategy):
    """Always generates HOLD decisions representing Always Flat benchmark."""
    def decide(self, features) -> Decision | None:
        return Decision(action="HOLD", confidence=1.0, reason="Always Flat benchmark")


@StrategyRegistry.register("random_predictor")
class RandomPredictorStrategy(BaseStrategy):
    """Generates random decisions (BUY, SELL, HOLD) for benchmark comparison."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def decide(self, features) -> Decision | None:
        action = self.rng.choice(["BUY", "SELL", "HOLD"], p=[0.33, 0.33, 0.34])
        return Decision(action=action, confidence=0.5, reason=f"Random prediction benchmark ({action})")


@StrategyRegistry.register("ema_crossover")
class EmaCrossoverStrategy(BaseStrategy):
    """Crossover logic: BUY when close > ema20, SELL when close <= ema20."""
    def decide(self, features) -> Decision | None:
        if features is None:
            return None
        action = "BUY" if features.close > features.ema20 else "SELL"
        return Decision(action=action, confidence=0.7, reason=f"EMA crossover: close is {'above' if action == 'BUY' else 'below'} ema20")


@StrategyRegistry.register("rsi_strategy")
class RsiStrategy(BaseStrategy):
    """Classical RSI strategy: BUY if <= 30, SELL if >= 70, else HOLD."""
    def decide(self, features) -> Decision | None:
        if features is None:
            return None
        if features.rsi <= 30:
            return Decision(action="BUY", confidence=0.8, reason="RSI oversold indicator")
        elif features.rsi >= 70:
            return Decision(action="SELL", confidence=0.8, reason="RSI overbought indicator")
        return Decision(action="HOLD", confidence=0.5, reason="RSI neutral zone")


@StrategyRegistry.register("macd_strategy")
class MacdStrategy(BaseStrategy):
    """MACD crossover: BUY if histogram > 0, SELL if histogram < 0, else HOLD."""
    def decide(self, features) -> Decision | None:
        if features is None:
            return None
        if features.macd_histogram > 0:
            return Decision(action="BUY", confidence=0.75, reason="MACD histogram bullish")
        elif features.macd_histogram < 0:
            return Decision(action="SELL", confidence=0.75, reason="MACD histogram bearish")
        return Decision(action="HOLD", confidence=0.5, reason="MACD neutral")
