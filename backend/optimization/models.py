from typing import Dict, Any, List
from backend.backtest.walk_forward import WalkForwardResult


class ParameterSet:

    def __init__(self, values: Dict[str, Any]):
        self.values = values

    def __repr__(self) -> str:
        return f"ParameterSet({self.values})"

    def to_dict(self) -> Dict[str, Any]:
        return self.values


class OptimizationResult:

    def __init__(
        self,
        parameter_set: ParameterSet,
        metrics: Dict[str, Any],
        walk_forward_result: WalkForwardResult | None = None,
    ):
        self.parameter_set = parameter_set
        self.metrics = metrics
        self.walk_forward_result = walk_forward_result


class OptimizationRun:

    def __init__(
        self,
        strategy_name: str,
        parameter_definitions: Dict[str, Any],
        optimization_config: Dict[str, Any],
        results: List[OptimizationResult],
    ):
        self.strategy_name = strategy_name
        self.parameter_definitions = parameter_definitions
        self.optimization_config = optimization_config
        self.results = results
