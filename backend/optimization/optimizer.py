import inspect
import itertools
from typing import Dict, Any, List, Sequence
from backend.models.candle import Candle
from backend.strategy.registry import StrategyRegistry
from backend.backtest.walk_forward import WalkForwardEngine
from backend.optimization.models import ParameterSet, OptimizationResult, OptimizationRun


def validate_parameters(strategy_name: str, parameter_definitions: Dict[str, Any]):
    try:
        strategy_cls = StrategyRegistry.get_strategy_class(strategy_name)
    except ValueError as e:
        raise ValueError(f"Invalid strategy: {strategy_name}") from e

    if not isinstance(parameter_definitions, dict):
        raise ValueError("Parameter definitions must be a dictionary.")

    sig = inspect.signature(strategy_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}

    for param_name, values in parameter_definitions.items():
        if param_name not in valid_params:
            raise ValueError(
                f"Param '{param_name}' is not accepted by strategy '{strategy_name}'."
            )
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(
                f"Values for parameter '{param_name}' must be an iterable list."
            )
        if len(values) == 0:
            raise ValueError(
                f"Values list for parameter '{param_name}' cannot be empty."
            )


def generate_parameter_grid(
    parameter_definitions: Dict[str, List[Any]]
) -> List[ParameterSet]:
    if not parameter_definitions:
        return [ParameterSet({})]

    keys = list(parameter_definitions.keys())
    values_list = list(parameter_definitions.values())

    combinations = list(itertools.product(*values_list))

    parameter_sets = []
    seen = set()
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        frozen_vals = tuple(sorted(param_dict.items()))
        if frozen_vals not in seen:
            seen.add(frozen_vals)
            parameter_sets.append(ParameterSet(param_dict))

    return parameter_sets


class ParameterOptimizer:

    def __init__(self, wfa_engine: WalkForwardEngine | None = None, repo=None):
        self.wfa_engine = (
            wfa_engine if wfa_engine is not None else WalkForwardEngine()
        )
        from backend.experiment.repository import LocalJsonExperimentRepository

        self.repo = (
            repo if repo is not None else LocalJsonExperimentRepository()
        )

    def optimize(
        self,
        strategy_name: str,
        parameter_definitions: Dict[str, Any],
        optimization_config: Dict[str, Any],
        candles: Sequence[Candle],
    ) -> OptimizationRun:
        validate_parameters(strategy_name, parameter_definitions)

        parameter_sets = generate_parameter_grid(parameter_definitions)

        train_days = optimization_config.get("train_days", 10)
        test_days = optimization_config.get("test_days", 5)
        step_days = optimization_config.get("step_days", 5)
        sorting_metric = optimization_config.get("sorting_metric", "net_profit")
        sorting_reverse = optimization_config.get("sorting_reverse", True)

        results = []

        for param_set in parameter_sets:
            wfa_result = self.wfa_engine.run(
                candles=candles,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                strategy_name=strategy_name,
                strategy_params=param_set.to_dict(),
            )

            metrics = wfa_result.global_stats
            results.append(
                OptimizationResult(
                    parameter_set=param_set,
                    metrics=metrics,
                    walk_forward_result=wfa_result,
                )
            )

        def get_sort_key(res: OptimizationResult):
            val = res.metrics.get(sorting_metric, 0.0)
            return val if val is not None else 0.0

        results.sort(key=get_sort_key, reverse=sorting_reverse)

        run = OptimizationRun(
            strategy_name=strategy_name,
            parameter_definitions=parameter_definitions,
            optimization_config=optimization_config,
            results=results,
        )

        # Build and persist experiment
        import uuid
        import datetime
        from backend.experiment.git_helper import get_git_commit_hash
        from backend.experiment.models import (
            Experiment,
            ExperimentRun,
            ExperimentSummary,
        )

        if run.results:
            best_result = run.results[0]
            best_metrics = best_result.metrics
            best_params = best_result.parameter_set.to_dict()
        else:
            best_metrics = {}
            best_params = {}

        exp_runs = []
        for res in run.results:
            exp_runs.append(
                ExperimentRun(
                    parameter_set=res.parameter_set.to_dict(),
                    metrics=res.metrics,
                )
            )

        avg_metrics = {}
        if run.results:
            metrics_to_avg = [
                "win_rate",
                "net_profit",
                "gross_profit",
                "gross_loss",
                "profit_factor",
                "maximum_drawdown_pct",
            ]
            for m in metrics_to_avg:
                vals = [
                    r.metrics.get(m, 0.0)
                    for r in run.results
                    if r.metrics.get(m) is not None
                ]
                avg_metrics[f"avg_{m}"] = (
                    sum(vals) / len(vals) if vals else 0.0
                )

        summary = ExperimentSummary(
            total_runs=len(run.results),
            best_run_parameters=best_params,
            best_run_metrics=best_metrics,
            aggregated_metrics=avg_metrics,
        )

        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        experiment_id = str(uuid.uuid4())
        experiment_name = f"Opt_{strategy_name}_{timestamp_str.replace(':', '-')}"

        symbols = (
            list(set(c.symbol for c in candles)) if candles else ["unknown"]
        )
        timeframe = candles[0].timeframe if candles else "unknown"

        experiment = Experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            timestamp=timestamp_str,
            git_commit_hash=get_git_commit_hash(),
            strategy_name=strategy_name,
            strategy_parameters=parameter_definitions,
            optimization_parameters=optimization_config,
            dataset_identifier=f"{len(candles)}_candles" if candles else "empty",
            timeframe=timeframe,
            symbols=symbols,
            training_window=train_days,
            testing_window=test_days,
            walk_forward_config=optimization_config,
            performance_metrics=best_metrics,
            notes="Auto-generated from ParameterOptimizer run",
            runs=exp_runs,
            summary=summary,
        )

        self.repo.save(experiment)

        return run
