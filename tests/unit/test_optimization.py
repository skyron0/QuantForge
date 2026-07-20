import pytest
from backend.optimization.optimizer import (
    validate_parameters,
    generate_parameter_grid,
    ParameterOptimizer,
)
from backend.optimization.models import ParameterSet


def test_validate_parameters_correct():
    """
    Verify valid parameter definitions pass without error.
    """
    # buy_threshold and sell_threshold are valid constructor parameters for RuleBasedStrategy
    defs = {
        "buy_threshold": [50.0, 60.0],
        "sell_threshold": [-50.0, -40.0],
    }
    validate_parameters("rule_based", defs)


def test_validate_parameters_invalid_name():
    """
    Verify invalid parameter names raise ValueError.
    """
    defs = {
        "invalid_param_name": [50.0],
    }
    with pytest.raises(ValueError) as exc:
        validate_parameters("rule_based", defs)
    # Check that error describes parameter discrepancy
    assert "is not accepted by strategy" in str(exc.value)


def test_validate_parameters_invalid_values_type():
    """
    Verify parameter values that are not list/tuple/set raise ValueError.
    """
    defs = {
        "buy_threshold": 50.0,  # Not iterable list
    }
    with pytest.raises(ValueError) as exc:
        validate_parameters("rule_based", defs)
    assert "must be an iterable list" in str(exc.value)


def test_validate_parameters_empty_values():
    """
    Verify empty list parameter values raise ValueError.
    """
    defs = {
        "buy_threshold": [],
    }
    with pytest.raises(ValueError) as exc:
        validate_parameters("rule_based", defs)
    assert "cannot be empty" in str(exc.value)


def test_validate_parameters_unknown_strategy():
    """
    Verify unknown strategy name raises ValueError.
    """
    with pytest.raises(ValueError) as exc:
        validate_parameters("non_existent_strategy", {"buy_threshold": [50]})
    assert "Invalid strategy" in str(exc.value)


def test_generate_parameter_grid_combinations():
    """
    Verify grid combinations are generated cleanly.
    """
    defs = {
        "buy_threshold": [50, 60],
        "sell_threshold": [-40],
    }
    grid = generate_parameter_grid(defs)
    assert len(grid) == 2
    # Verify sets contain correct mapping
    vals = [p.to_dict() for p in grid]
    assert {"buy_threshold": 50, "sell_threshold": -40} in vals
    assert {"buy_threshold": 60, "sell_threshold": -40} in vals


def test_generate_parameter_grid_deduplication():
    """
    Verify duplicate inputs are automatically filtered out.
    """
    defs = {
        "buy_threshold": [50, 50, 60],  # Duplicate 50
        "sell_threshold": [-40, -40],  # Duplicate -40
    }
    grid = generate_parameter_grid(defs)
    assert len(grid) == 2  # Combination should yield only 2 unique sets: {50,-40} and {60,-40}


def test_optimizer_runs_and_ranks():
    """
    Verify ParameterOptimizer generates combination runs and ranks results correctly using a mock WFA.
    """
    # Create fake WFA engine
    class MockWFAEngine:
        def run(
            self,
            candles,
            train_days,
            test_days,
            step_days,
            strategy_name,
            strategy_params,
        ):
            buy = strategy_params.get("buy_threshold", 0.0)
            sell = strategy_params.get("sell_threshold", 0.0)
            net_profit = buy * 2 + sell

            class FakeWFA:
                global_stats = {
                    "net_profit": net_profit,
                    "win_rate": 55.0,
                }
            return FakeWFA()

    optimizer = ParameterOptimizer(wfa_engine=MockWFAEngine())  # type: ignore

    defs = {
        "buy_threshold": [50.0, 60.0],
        "sell_threshold": [-40.0],
    }
    config = {
        "train_days": 10,
        "test_days": 5,
        "step_days": 5,
        "sorting_metric": "net_profit",
        "sorting_reverse": True,  # Descending
    }

    # Combos:
    # 1. buy=50, sell=-40 -> profit = 100 - 40 = 60.0
    # 2. buy=60, sell=-40 -> profit = 120 - 40 = 80.0
    # Since reverse=True, combo 2 (profit 80) must rank first!
    run = optimizer.optimize("rule_based", defs, config, candles=[])

    assert run.strategy_name == "rule_based"
    assert len(run.results) == 2

    # Best first
    best = run.results[0]
    worst = run.results[1]

    assert best.parameter_set.to_dict()["buy_threshold"] == 60.0
    assert best.metrics["net_profit"] == 80.0

    assert worst.parameter_set.to_dict()["buy_threshold"] == 50.0
    assert worst.metrics["net_profit"] == 60.0
