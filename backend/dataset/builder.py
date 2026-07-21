import os
import json
import datetime
import uuid
from typing import Sequence, Dict, Any, Tuple

import pandas as pd

from backend.models.candle import Candle
from backend.indicator.indicator_engine import IndicatorEngine
from backend.feature.feature_engine import FeatureEngine
from backend.dataset.labelers import BaseLabelStrategy
from backend.dataset.models import DatasetMetadata


class DatasetBuilder:

    def __init__(
        self,
        labeler: BaseLabelStrategy,
        label_params: Dict[str, Any] | None = None,
        output_dir: str = "data/datasets",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.labeler = labeler
        self.label_params = label_params or {}
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.indicator_engine = IndicatorEngine()
        self.feature_engine = FeatureEngine()

    def build(
        self,
        candles: Sequence[Candle],
        dataset_name: str = "default",
        export_csv: bool = False,
    ) -> Tuple[DatasetMetadata, pd.DataFrame]:
        rows = []
        feature_list = None

        for i in range(len(candles)):
            # We need enough candles up to index i for indicators
            window = candles[: i + 1]
            indicators = self.indicator_engine.calculate(window)
            if indicators is None:
                continue

            fv = self.feature_engine.build(candles[i], indicators)
            label = self.labeler.label(candles, i, self.label_params)
            if label is None:
                continue

            row = {
                "symbol": fv.symbol,
                "timestamp": str(candles[i].open_time),
                "close": fv.close,
                "rsi": fv.rsi,
                "ema20": fv.ema20,
                "macd": fv.macd,
                "macd_signal": fv.macd_signal,
                "macd_histogram": fv.macd_histogram,
                "adx": fv.adx,
                "atr": fv.atr,
                "bb_upper": fv.bb_upper,
                "bb_middle": fv.bb_middle,
                "bb_lower": fv.bb_lower,
                "vwap": fv.vwap,
                "label": label,
            }
            rows.append(row)

            if feature_list is None:
                feature_list = [
                    k for k in row.keys() if k not in ("symbol", "timestamp", "label")
                ]

        df = pd.DataFrame(rows)

        # Split
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        df["split"] = "test"
        df.iloc[:train_end, df.columns.get_loc("split")] = "train"
        df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"

        # Metadata
        symbols = list(df["symbol"].unique()) if not df.empty else []
        timeframe = candles[0].timeframe if candles else "unknown"

        dataset_id = str(uuid.uuid4())
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            generation_timestamp=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            source_symbols=symbols,
            timeframe=timeframe,
            feature_version="v1",
            label_version="v1",
            generation_parameters=self.label_params,
            sample_count=n,
            feature_list=feature_list or [],
            labeling_strategy=self.labeler.name,
            train_count=train_end,
            val_count=val_end - train_end,
            test_count=n - val_end,
            label_horizon=self.label_params.get("horizon", 1),
        )

        # Persist
        os.makedirs(self.output_dir, exist_ok=True)
        version_dir = os.path.join(self.output_dir, f"{dataset_name}_{dataset_id}")
        os.makedirs(version_dir, exist_ok=True)

        parquet_path = os.path.join(version_dir, "dataset.parquet")
        df.to_parquet(parquet_path, index=False)

        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=4, ensure_ascii=False)

        if export_csv:
            csv_path = os.path.join(version_dir, "dataset.csv")
            df.to_csv(csv_path, index=False)

        return metadata, df
