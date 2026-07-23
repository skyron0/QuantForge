import threading
from typing import Any, Dict, List


class MarketDataTelemetrySink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: Dict[str, int] = {
            "received_messages": 0,
            "accepted_messages": 0,
            "validation_failures": 0,
            "sequence_gaps": 0,
            "duplicate_messages": 0,
            "out_of_order_messages": 0,
            "snapshots_built": 0,
            "snapshot_failures": 0,
        }
        self._latency_samples: List[float] = []

    def record_received(self) -> None:
        with self._lock:
            self._metrics["received_messages"] += 1

    def record_accepted(self) -> None:
        with self._lock:
            self._metrics["accepted_messages"] += 1

    def record_validation_failure(self) -> None:
        with self._lock:
            self._metrics["validation_failures"] += 1

    def record_sequence_gap(self) -> None:
        with self._lock:
            self._metrics["sequence_gaps"] += 1

    def record_duplicate(self) -> None:
        with self._lock:
            self._metrics["duplicate_messages"] += 1

    def record_out_of_order(self) -> None:
        with self._lock:
            self._metrics["out_of_order_messages"] += 1

    def record_snapshot_built(self) -> None:
        with self._lock:
            self._metrics["snapshots_built"] += 1

    def record_snapshot_failure(self) -> None:
        with self._lock:
            self._metrics["snapshot_failures"] += 1

    def record_latency(self, latency_seconds: float) -> None:
        with self._lock:
            self._latency_samples.append(latency_seconds)
            # Bound memory of latency samples
            if len(self._latency_samples) > 1000:
                self._latency_samples.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = (
                sum(self._latency_samples) / len(self._latency_samples)
                if self._latency_samples
                else 0.0
            )
            return {
                **self._metrics,
                "average_ingestion_latency_seconds": avg_latency,
                "latency_sample_count": len(self._latency_samples)
            }


class ConsoleMarketDataTelemetrySink(MarketDataTelemetrySink):
    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose

    def record_validation_failure(self) -> None:
        super().record_validation_failure()
        if self.verbose:
            print("[MarketDataTelemetry] WARN: Validation failure recorded.")

    def record_sequence_gap(self) -> None:
        super().record_sequence_gap()
        print("[MarketDataTelemetry] ALERT: Sequence gap detected!")

    def record_duplicate(self) -> None:
        super().record_duplicate()
        if self.verbose:
            print("[MarketDataTelemetry] INFO: Duplicate message filtered.")

    def record_out_of_order(self) -> None:
        super().record_out_of_order()
        print("[MarketDataTelemetry] WARN: Out-of-order message filtered or rejected.")
