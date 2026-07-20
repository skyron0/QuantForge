import os
import json
from abc import ABC, abstractmethod
from typing import List, Optional
from backend.experiment.models import Experiment


class ExperimentRepository(ABC):

    @abstractmethod
    def save(self, experiment: Experiment) -> None:
        pass

    @abstractmethod
    def get(self, experiment_id: str) -> Optional[Experiment]:
        pass

    @abstractmethod
    def list(self) -> List[Experiment]:
        pass

    @abstractmethod
    def delete(self, experiment_id: str) -> None:
        pass

    @abstractmethod
    def exists(self, experiment_id: str) -> bool:
        pass


class LocalJsonExperimentRepository(ExperimentRepository):

    def __init__(self, directory: str = "data/experiments"):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _get_path(self, experiment_id: str) -> str:
        return os.path.join(self.directory, f"experiment_{experiment_id}.json")

    def save(self, experiment: Experiment) -> None:
        path = self._get_path(experiment.experiment_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, indent=4, ensure_ascii=False)

    def get(self, experiment_id: str) -> Optional[Experiment]:
        path = self._get_path(experiment_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Experiment.from_dict(data)

    def list(self) -> List[Experiment]:
        experiments = []
        for filename in os.listdir(self.directory):
            if filename.startswith("experiment_") and filename.endswith(".json"):
                path = os.path.join(self.directory, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        experiments.append(Experiment.from_dict(data))
                except Exception:
                    pass
        return experiments

    def delete(self, experiment_id: str) -> None:
        path = self._get_path(experiment_id)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, experiment_id: str) -> bool:
        path = self._get_path(experiment_id)
        return os.path.exists(path)
