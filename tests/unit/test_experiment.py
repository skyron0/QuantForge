import pytest
import os
import shutil
import tempfile
from backend.experiment.models import (
    Experiment,
    ExperimentRun,
    ExperimentArtifact,
    ExperimentSummary,
)
from backend.experiment.repository import LocalJsonExperimentRepository
from backend.experiment.git_helper import get_git_commit_hash
from backend.optimization.optimizer import ParameterOptimizer


@pytest.fixture
def temp_dir():
    # Setup temporary directory for repository testing
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_models_to_from_dict():
    """
    Verify complete model serialization cycle.
    """
    artifact = ExperimentArtifact(
        name="test_plot",
        artifact_type="plot",
        path="plots/p.png",
        content_summary="Test plot summary",
    )
    run = ExperimentRun(
        parameter_set={"p": 12},
        metrics={"net_profit": 150.0},
        artifacts=[artifact],
    )
    summary = ExperimentSummary(
        total_runs=1,
        best_run_parameters={"p": 12},
        best_run_metrics={"net_profit": 150.0},
        aggregated_metrics={"avg_net_profit": 150.0},
    )
    experiment = Experiment(
        experiment_id="exp-123",
        experiment_name="MyExp",
        timestamp="2026-07-20T12:00:00Z",
        git_commit_hash="abcdef",
        strategy_name="rule_based",
        strategy_parameters={"buy_threshold": [50]},
        optimization_parameters={"train_days": 10},
        dataset_identifier="candles_btc",
        timeframe="5m",
        symbols=["BTCUSD"],
        training_window=10,
        testing_window=5,
        walk_forward_config={"train_days": 10},
        performance_metrics={"net_profit": 150.0},
        notes="Notes field",
        runs=[run],
        artifacts=[artifact],
        summary=summary,
    )

    data = experiment.to_dict()
    restored = Experiment.from_dict(data)

    assert restored.experiment_id == "exp-123"
    assert restored.experiment_name == "MyExp"
    assert restored.git_commit_hash == "abcdef"
    assert len(restored.runs) == 1
    assert restored.runs[0].parameter_set == {"p": 12}
    assert len(restored.runs[0].artifacts) == 1
    assert restored.runs[0].artifacts[0].name == "test_plot"
    assert restored.summary is not None
    assert restored.summary.total_runs == 1


def test_git_commit_hash_resolution():
    """
    Verify git commit resolution runs without errors.
    """
    sha = get_git_commit_hash()
    # It should be either a valid 40-char SHA string or None (if git command fails/unavailable)
    if sha is not None:
        assert isinstance(sha, str)
        assert len(sha) == 40 or len(sha) == 7  # typical sizes


def test_repository_lifecycle(temp_dir):
    """
    Verify LocalJsonExperimentRepository save, get, list, exists, and delete lifecycle.
    """
    repo = LocalJsonExperimentRepository(directory=temp_dir)
    experiment = Experiment(
        experiment_id="exp-x",
        experiment_name="ExpX",
        timestamp="2026-07-20T12:00:00Z",
        git_commit_hash=None,
        strategy_name="rule_based",
        strategy_parameters={},
        optimization_parameters={},
        dataset_identifier="candles",
        timeframe="5m",
        symbols=["BTC"],
        training_window=10,
        testing_window=5,
        walk_forward_config={},
        performance_metrics={},
        notes="",
    )

    assert not repo.exists("exp-x")
    repo.save(experiment)
    assert repo.exists("exp-x")

    restored = repo.get("exp-x")
    assert restored is not None
    assert restored.experiment_name == "ExpX"

    all_experiments = repo.list()
    assert len(all_experiments) == 1
    assert all_experiments[0].experiment_id == "exp-x"

    repo.delete("exp-x")
    assert not repo.exists("exp-x")
    assert repo.get("exp-x") is None


def test_optimizer_integration(temp_dir):
    """
    Verify ParameterOptimizer automatically saves an experiment JSON file upon optimizing.
    """
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
            class FakeWFA:
                global_stats = {"net_profit": 100.0, "win_rate": 60.0}
            return FakeWFA()

    repo = LocalJsonExperimentRepository(directory=temp_dir)
    optimizer = ParameterOptimizer(wfa_engine=MockWFAEngine(), repo=repo)  # type: ignore

    defs = {
        "buy_threshold": [50.0],
        "sell_threshold": [-40.0],
    }
    config = {
        "train_days": 10,
        "test_days": 5,
        "step_days": 5,
        "sorting_metric": "net_profit",
    }

    # Run optimizer
    assert len(repo.list()) == 0
    optimizer.optimize("rule_based", defs, config, candles=[])

    # Check repository contains the newly tracked experiment
    experiments = repo.list()
    assert len(experiments) == 1
    exp = experiments[0]

    assert exp.strategy_name == "rule_based"
    assert exp.strategy_parameters == defs
    assert exp.optimization_parameters == config
    assert len(exp.runs) == 1
    assert exp.runs[0].parameter_set == {"buy_threshold": 50.0, "sell_threshold": -40.0}
    assert exp.runs[0].metrics["net_profit"] == 100.0

    # Verify summary calculations
    assert exp.summary is not None
    assert exp.summary.total_runs == 1
    assert exp.summary.best_run_parameters == {"buy_threshold": 50.0, "sell_threshold": -40.0}
    assert exp.summary.best_run_metrics["net_profit"] == 100.0
    assert exp.summary.aggregated_metrics["avg_net_profit"] == 100.0
    assert exp.summary.aggregated_metrics["avg_win_rate"] == 60.0
