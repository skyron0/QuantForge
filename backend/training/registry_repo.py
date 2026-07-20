import os
import json
from typing import Dict, Any, List, Optional

from backend.training.registry_models import RegisteredModel


class LocalModelRegistryRepository:
    """JSON-based repository to persist candidate or validated models."""

    def __init__(self, db_path: str = "data/registry/model_registry.json"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _read_all(self) -> Dict[str, Any]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _write_all(self, data: Dict[str, Any]) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def save(self, model: RegisteredModel) -> None:
        data = self._read_all()
        data[model.model_version] = model.to_dict()
        self._write_all(data)

    def get(self, model_version: str) -> Optional[RegisteredModel]:
        data = self._read_all()
        if model_version not in data:
            return None
        return RegisteredModel.from_dict(data[model_version])

    def list_all(self) -> List[RegisteredModel]:
        data = self._read_all()
        return [RegisteredModel.from_dict(v) for v in data.values()]

    def delete(self, model_version: str) -> None:
        data = self._read_all()
        if model_version in data:
            del data[model_version]
            self._write_all(data)
