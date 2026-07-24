import abc
import csv
import os
from decimal import Decimal
from typing import List, Dict, Any
from backend.replay.exceptions import DatasetLoadingError
from backend.replay.dataset import validate_ohlc_record, sort_records_deterministically

class HistoricalDatasetLoader(abc.ABC):
    """
    Abstract base interface for historical data loaders.
    """
    @abc.abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Loads and returns a list of normalized market data records."""
        pass

class CSVHistoricalCandleLoader(HistoricalDatasetLoader):
    """
    Concrete loader mapping historical CSV candles to domain representation.
    Expects headers: timestamp/open_time, open, high, low, close, volume (optional).
    """
    def __init__(self, file_path: str, symbol: str, timeframe: str = "1m") -> None:
        self.file_path = file_path
        self.symbol = symbol.upper().replace("/", "")
        self.timeframe = timeframe

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            raise DatasetLoadingError(f"Dataset file not found: {self.file_path}")

        records: List[Dict[str, Any]] = []
        try:
            with open(self.file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise DatasetLoadingError(f"CSV file is empty or missing headers: {self.file_path}")

                # Normalize headers to lowercase
                headers = [h.strip().lower() for h in reader.fieldnames]
                
                # Determine timestamp column name
                time_col = None
                for col in ["timestamp", "open_time", "open time", "time", "date"]:
                    if col in headers:
                        # Find the actual case-sensitive header
                        time_col = reader.fieldnames[headers.index(col)]
                        break
                
                if not time_col:
                    raise DatasetLoadingError(
                        f"CSV must contain a timestamp/open_time column. Headers: {reader.fieldnames}"
                    )

                # Map price/volume headers
                field_map = {}
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in headers:
                        field_map[col] = reader.fieldnames[headers.index(col)]
                    elif col != "volume":
                        raise DatasetLoadingError(f"Missing required column in CSV: {col}")

                sequence_idx = 0
                for row_idx, row in enumerate(reader, start=2):
                    try:
                        timestamp_str = row[time_col].strip()
                        if not timestamp_str:
                            continue

                        # Read required fields
                        op = row[field_map["open"]].strip()
                        hi = row[field_map["high"]].strip()
                        lo = row[field_map["low"]].strip()
                        cl = row[field_map["close"]].strip()
                        
                        vol = "0"
                        if "volume" in field_map:
                            vol = row[field_map["volume"]].strip() or "0"

                        record = {
                            "symbol": self.symbol,
                            "timeframe": self.timeframe,
                            "open_time": timestamp_str,  # Match domain Candle schema
                            "timestamp": timestamp_str,
                            "open": op,
                            "high": hi,
                            "low": lo,
                            "close": cl,
                            "volume": vol,
                            "sequence": sequence_idx,
                            "source": "csv_replay",
                            "closed": True
                        }
                        
                        # Apply numeric value/OHLC validation
                        validate_ohlc_record(record)
                        
                        records.append(record)
                        sequence_idx += 1

                    except KeyError as e:
                        raise DatasetLoadingError(f"Missing column {str(e)} in row {row_idx} of CSV: {self.file_path}")
                    except Exception as e:
                        raise DatasetLoadingError(f"Malformed data at row {row_idx}: {str(e)}") from e

        except Exception as e:
            if isinstance(e, DatasetLoadingError):
                raise e
            raise DatasetLoadingError(f"Failed to read CSV dataset: {self.file_path}. Details: {str(e)}") from e

        if not records:
            raise DatasetLoadingError(f"No valid records loaded from dataset: {self.file_path}")

        # Deterministically sort the records
        return sort_records_deterministically(records)
