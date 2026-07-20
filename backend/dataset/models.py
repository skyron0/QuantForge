from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class DatasetMetadata:
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation_timestamp: str = ""
    source_symbols: List[str] = field(default_factory=list)
    timeframe: str = ""
    feature_version: str = "v1"
    label_version: str = "v1"
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    feature_list: List[str] = field(default_factory=list)
    labeling_strategy: str = ""
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "generation_timestamp": self.generation_timestamp,
            "source_symbols": self.source_symbols,
            "timeframe": self.timeframe,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "generation_parameters": self.generation_parameters,
            "sample_count": self.sample_count,
            "feature_list": self.feature_list,
            "labeling_strategy": self.labeling_strategy,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetMetadata":
        return cls(
            dataset_id=d["dataset_id"],
            generation_timestamp=d["generation_timestamp"],
            source_symbols=d["source_symbols"],
            timeframe=d["timeframe"],
            feature_version=d.get("feature_version", "v1"),
            label_version=d.get("label_version", "v1"),
            generation_parameters=d["generation_parameters"],
            sample_count=d["sample_count"],
            feature_list=d["feature_list"],
            labeling_strategy=d["labeling_strategy"],
            train_count=d.get("train_count", 0),
            val_count=d.get("val_count", 0),
            test_count=d.get("test_count", 0),
        )
