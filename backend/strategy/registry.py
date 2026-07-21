import importlib
from typing import Dict
from typing import Type
from backend.strategy.base import BaseStrategy


class StrategyRegistry:

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a BaseStrategy subclass.
        """
        def decorator(subclass: Type[BaseStrategy]):
            cls._strategies[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_strategy_class(cls, name: str) -> Type[BaseStrategy]:
        """
        Retrieve a registered strategy class by name.
        """
        strategy_cls = cls._strategies.get(name.lower())
        if not strategy_cls:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return strategy_cls

    @classmethod
    def list_strategies(cls):
        """
        List all registered strategy names.
        """
        return list(cls._strategies.keys())


class StrategyLoader:

    @staticmethod
    def load_strategy(name: str, **kwargs) -> BaseStrategy:
        """
        Load a strategy instance dynamically, importing its module if necessary.
        """
        # Ensure our validation benchmarks and prediction frameworks are registered
        for mod in ["backend.strategy.benchmarks", "backend.strategy.prediction_strategy"]:
            try:
                importlib.import_module(mod)
            except ImportError:
                pass

        try:
            strategy_cls = StrategyRegistry.get_strategy_class(name)
        except ValueError:
            # If not found, try importing built-in strategy module
            try:
                importlib.import_module(f"backend.strategy.{name.lower()}")
            except ImportError:
                pass
            strategy_cls = StrategyRegistry.get_strategy_class(name)

        return strategy_cls(**kwargs)
