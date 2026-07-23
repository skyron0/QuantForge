"""
Telemetry sinks for the Feature Runtime pipeline.

Tracks warmup skips, invalid snapshots, schema mismatches,
inference calls/failures, signals generated, neutral signals,
stage and total latency.
"""

import threading
from typing import Dict, List, Optional

from backend.feature_runtime.models import FeatureRuntimeResult, FeatureRuntimeStatus


class FeatureRuntimeTelemetrySink:
    """Abstract telemetry sink — subclass to customise output."""

    def record_result(self, result: FeatureRuntimeResult) -> None:
        """Record a completed pipeline result."""
        pass

    def get_metrics(self) -> Dict[str, object]:
        """Return accumulated metrics snapshot."""
        return {}


class ConsoleFeatureRuntimeTelemetrySink(FeatureRuntimeTelemetrySink):
    """
    Thread-safe, in-memory telemetry sink that accumulates counters and
    latency records.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.warmup_skips: int = 0
        self.invalid_snapshots: int = 0
        self.schema_mismatches: int = 0
        self.inference_calls: int = 0
        self.inference_failures: int = 0
        self.signals_generated: int = 0
        self.neutral_signals: int = 0
        self.stale_data_skips: int = 0
        self.total_results: int = 0
        self.stage_latencies: Dict[str, List[float]] = {}
        self.total_latencies: List[float] = []

    def record_result(self, result: FeatureRuntimeResult) -> None:
        with self._lock:
            self.total_results += 1
            self.total_latencies.append(result.total_latency_ms)

            for stage, ms in result.stage_timings.items():
                if stage not in self.stage_latencies:
                    self.stage_latencies[stage] = []
                self.stage_latencies[stage].append(ms)

            if result.status == FeatureRuntimeStatus.WARMUP_SKIP:
                self.warmup_skips += 1
            elif result.status == FeatureRuntimeStatus.VALIDATION_FAILED:
                self.invalid_snapshots += 1
            elif result.status == FeatureRuntimeStatus.SCHEMA_MISMATCH:
                self.schema_mismatches += 1
            elif result.status == FeatureRuntimeStatus.STALE_DATA:
                self.stale_data_skips += 1
            elif result.status == FeatureRuntimeStatus.INFERENCE_FAILED:
                self.inference_failures += 1
                self.inference_calls += 1
            elif result.status == FeatureRuntimeStatus.SUCCESS:
                self.inference_calls += 1
                self.signals_generated += 1
                if result.ml_signal and result.ml_signal.direction == "NEUTRAL":
                    self.neutral_signals += 1
            elif result.status == FeatureRuntimeStatus.SIGNAL_FAILED:
                self.inference_calls += 1

    def get_metrics(self) -> Dict[str, object]:
        with self._lock:
            return {
                "total_results": self.total_results,
                "warmup_skips": self.warmup_skips,
                "invalid_snapshots": self.invalid_snapshots,
                "schema_mismatches": self.schema_mismatches,
                "stale_data_skips": self.stale_data_skips,
                "inference_calls": self.inference_calls,
                "inference_failures": self.inference_failures,
                "signals_generated": self.signals_generated,
                "neutral_signals": self.neutral_signals,
                "total_latencies_count": len(self.total_latencies),
                "avg_total_latency_ms": (
                    sum(self.total_latencies) / len(self.total_latencies)
                    if self.total_latencies
                    else 0.0
                ),
            }
