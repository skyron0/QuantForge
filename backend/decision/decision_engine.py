from backend.strategy.base import BaseStrategy
from backend.strategy.registry import StrategyLoader
from backend.decision.models import Decision


class DecisionEngine:

    def __init__(self, strategy: BaseStrategy | None = None):
        if strategy is None:
            self.strategy = StrategyLoader.load_strategy("rule_based")
        else:
            self.strategy = strategy

    def decide(self, features) -> Decision | None:
        return self.strategy.decide(features)