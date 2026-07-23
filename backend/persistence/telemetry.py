import threading
from typing import Dict, Any, List


class PersistenceTelemetry:
    """Thread-safe telemetry sink for tracking database operations performance and errors."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._write_count = 0
        self._read_count = 0
        self._write_errors = 0
        self._read_errors = 0
        self._table_write_latencies: Dict[str, List[float]] = {}
        self._table_read_latencies: Dict[str, List[float]] = {}

    def record_write(self, entity_type: str, status: str, latency_ms: float = 0.0) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._write_count += 1
            if status == "failure":
                self._write_errors += 1
            if entity_type not in self._table_write_latencies:
                self._table_write_latencies[entity_type] = []
            self._table_write_latencies[entity_type].append(latency_ms)

    def record_read(self, entity_type: str, status: str, latency_ms: float = 0.0) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._read_count += 1
            if status == "failure":
                self._read_errors += 1
            if entity_type not in self._table_read_latencies:
                self._table_read_latencies[entity_type] = []
            self._table_read_latencies[entity_type].append(latency_ms)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "write_count": self._write_count,
                "read_count": self._read_count,
                "write_errors": self._write_errors,
                "read_errors": self._read_errors,
                "table_write_latencies": {k: list(v) for k, v in self._table_write_latencies.items()},
                "table_read_latencies": {k: list(v) for k, v in self._table_read_latencies.items()},
            }
