from typing import Dict, Any, List, Optional
import uuid


class ExperimentArtifact:

    def __init__(
        self,
        name: str,
        artifact_type: str,
        path: str,
        content_summary: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ):
        self.artifact_id = artifact_id or str(uuid.uuid4())
        self.name = name
        self.type = artifact_type
        self.path = path
        self.content_summary = content_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "content_summary": self.content_summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentArtifact":
        return cls(
            artifact_id=d["artifact_id"],
            name=d["name"],
            artifact_type=d["type"],
            path=d["path"],
            content_summary=d.get("content_summary"),
        )


class ExperimentRun:

    def __init__(
        self,
        parameter_set: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Optional[List[ExperimentArtifact]] = None,
        run_id: Optional[str] = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.parameter_set = parameter_set
        self.metrics = metrics
        self.artifacts = artifacts or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parameter_set": self.parameter_set,
            "metrics": self.metrics,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRun":
        return cls(
            run_id=d["run_id"],
            parameter_set=d["parameter_set"],
            metrics=d["metrics"],
            artifacts=[
                ExperimentArtifact.from_dict(a) for a in d.get("artifacts", [])
            ],
        )


class ExperimentSummary:

    def __init__(
        self,
        total_runs: int,
        best_run_parameters: Dict[str, Any],
        best_run_metrics: Dict[str, Any],
        aggregated_metrics: Dict[str, Any],
    ):
        self.total_runs = total_runs
        self.best_run_parameters = best_run_parameters
        self.best_run_metrics = best_run_metrics
        self.aggregated_metrics = aggregated_metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "best_run_parameters": self.best_run_parameters,
            "best_run_metrics": self.best_run_metrics,
            "aggregated_metrics": self.aggregated_metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentSummary":
        return cls(
            total_runs=d["total_runs"],
            best_run_parameters=d["best_run_parameters"],
            best_run_metrics=d["best_run_metrics"],
            aggregated_metrics=d["aggregated_metrics"],
        )


class Experiment:

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        timestamp: str,
        git_commit_hash: Optional[str],
        strategy_name: str,
        strategy_parameters: Dict[str, Any],
        optimization_parameters: Dict[str, Any],
        dataset_identifier: str,
        timeframe: str,
        symbols: List[str],
        training_window: Any,
        testing_window: Any,
        walk_forward_config: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        notes: str,
        runs: Optional[List[ExperimentRun]] = None,
        artifacts: Optional[List[ExperimentArtifact]] = None,
        summary: Optional[ExperimentSummary] = None,
    ):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.timestamp = timestamp
        self.git_commit_hash = git_commit_hash
        self.strategy_name = strategy_name
        self.strategy_parameters = strategy_parameters
        self.optimization_parameters = optimization_parameters
        self.dataset_identifier = dataset_identifier
        self.timeframe = timeframe
        self.symbols = symbols
        self.training_window = training_window
        self.testing_window = testing_window
        self.walk_forward_config = walk_forward_config
        self.performance_metrics = performance_metrics
        self.notes = notes
        self.runs = runs or []
        self.artifacts = artifacts or []
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "git_commit_hash": self.git_commit_hash,
            "strategy_name": self.strategy_name,
            "strategy_parameters": self.strategy_parameters,
            "optimization_parameters": self.optimization_parameters,
            "dataset_identifier": self.dataset_identifier,
            "timeframe": self.timeframe,
            "symbols": self.symbols,
            "training_window": self.training_window,
            "testing_window": self.testing_window,
            "walk_forward_config": self.walk_forward_config,
            "performance_metrics": self.performance_metrics,
            "notes": self.notes,
            "runs": [r.to_dict() for r in self.runs],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "summary": self.summary.to_dict() if self.summary else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        summary_val = d.get("summary")
        summary_obj = (
            ExperimentSummary.from_dict(summary_val) if summary_val else None
        )
        return cls(
            experiment_id=d["experiment_id"],
            experiment_name=d["experiment_name"],
            timestamp=d["timestamp"],
            git_commit_hash=d.get("git_commit_hash"),
            strategy_name=d["strategy_name"],
            strategy_parameters=d["strategy_parameters"],
            optimization_parameters=d["optimization_parameters"],
            dataset_identifier=d["dataset_identifier"],
            timeframe=d["timeframe"],
            symbols=d["symbols"],
            training_window=d["training_window"],
            testing_window=d["testing_window"],
            walk_forward_config=d["walk_forward_config"],
            performance_metrics=d["performance_metrics"],
            notes=d["notes"],
            runs=[ExperimentRun.from_dict(r) for r in d.get("runs", [])],
            artifacts=[
                ExperimentArtifact.from_dict(a) for a in d.get("artifacts", [])
            ],
            summary=summary_obj,
        )
