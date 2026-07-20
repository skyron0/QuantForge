import pytest
from collections import namedtuple
from backend.strategy.base import BaseStrategy
from backend.strategy.rule_based import RuleBasedStrategy
from backend.strategy.registry import StrategyRegistry, StrategyLoader
from backend.decision.decision_engine import DecisionEngine
from backend.decision.models import Decision

# Mock features for test inputs
MockFeatures = namedtuple(
    "MockFeatures",
    [
        "rsi",
        "macd_histogram",
        "adx",
        "close",
        "ema20",
        "vwap",
        "bb_lower",
        "bb_upper",
    ],
)


def test_base_strategy_cannot_be_instantiated():
    """
    Verify BaseStrategy is an abstract base class and cannot be instantiated.
    """
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore


def test_rule_based_strategy_decision_parity():
    """
    Verify RuleBasedStrategy evaluates score rules correctly.
    """
    strategy = RuleBasedStrategy(buy_threshold=50, sell_threshold=-50)

    # 1. Bullish scenario (RSI oversold + MACD bullish + Above EMA20 + Above VWAP)
    # RSI <= 30 (+30), macd_histogram > 0 (+20), close > ema20 (+10) -> score = 60 (Buy)
    bullish_features = MockFeatures(
        rsi=25.0,
        macd_histogram=1.5,
        adx=20.0,
        close=50000.0,
        ema20=49500.0,
        vwap=49800.0,
        bb_lower=49000.0,
        bb_upper=51000.0,
    )
    decision = strategy.decide(bullish_features)
    assert decision is not None
    assert decision.action == "BUY"
    assert decision.confidence > 0.50

    # 2. Bearish scenario (RSI overbought + Below EMA20 + Below VWAP)
    # RSI >= 70 (-30), macd_histogram < 0 (-20), close <= ema20 (-10) -> score = -60 (Sell)
    bearish_features = MockFeatures(
        rsi=75.0,
        macd_histogram=-0.5,
        adx=20.0,
        close=48000.0,
        ema20=49500.0,
        vwap=49000.0,
        bb_lower=47000.0,
        bb_upper=51000.0,
    )
    decision = strategy.decide(bearish_features)
    assert decision is not None
    assert decision.action == "SELL"

    # 3. Hold/Neutral scenario
    neutral_features = MockFeatures(
        rsi=50.0,
        macd_histogram=0.0,
        adx=20.0,
        close=49000.0,
        ema20=49000.0,
        vwap=49000.0,
        bb_lower=48000.0,
        bb_upper=50000.0,
    )
    decision = strategy.decide(neutral_features)
    assert decision is not None
    assert decision.action == "HOLD"


def test_strategy_registry_registration():
    """
    Verify custom strategies can be registered and retrieved.
    """

    @StrategyRegistry.register("test_mock")
    class MockStrategy(BaseStrategy):
        def decide(self, features) -> Decision | None:
            return Decision(action="HOLD", confidence=0.88, reason="Mock")

    # Get class from registry
    cls = StrategyRegistry.get_strategy_class("test_mock")
    assert cls == MockStrategy

    # Instantiate via registry
    instance = cls()
    assert isinstance(instance, BaseStrategy)
    decision = instance.decide(None)
    assert decision is not None
    assert decision.action == "HOLD"
    assert decision.reason == "Mock"


def test_strategy_loader_loads_successfully():
    """
    Verify StrategyLoader loads rule_based strategy.
    """
    strategy = StrategyLoader.load_strategy("rule_based")
    assert isinstance(strategy, RuleBasedStrategy)


def test_decision_engine_delegation():
    """
    Verify DecisionEngine delegates execution to the strategy class.
    """
    # Create custom mock strategy
    class ConstantBuyStrategy(BaseStrategy):
        def decide(self, features) -> Decision | None:
            return Decision(action="BUY", confidence=0.99, reason="Constant")

    strategy = ConstantBuyStrategy()
    engine = DecisionEngine(strategy=strategy)

    decision = engine.decide(None)
    assert decision is not None
    assert decision.action == "BUY"
    assert decision.reason == "Constant"


def test_invalid_strategy_fails_gracefully():
    """
    Verify invalid strategy name triggers a clean ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        StrategyLoader.load_strategy("non_existent_strategy_name")
    assert "is not registered" in str(exc_info.value)
