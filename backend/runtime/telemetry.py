import threading
from typing import List, Optional


class RuntimeTelemetry:
    """Thread-safe collector for runtime metrics and engine coordination stats."""
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        
        self._runtime_state: str = "INITIALIZED"
        self._started_at: Optional[str] = None
        self._scheduler_iterations: int = 0
        self._cycle_count: int = 0
        
        # Latency accumulators
        self._total_cycle_latency_ms: float = 0.0
        self._total_dispatch_latency_ms: float = 0.0
        self._dispatch_count: int = 0
        
        self._subscriber_count: int = 0
        self._queue_depth: int = 0
        self._failed_events: int = 0
        self._failed_handlers: int = 0
        self._runtime_errors: List[str] = []

    def transition_state(self, new_state: str, timestamp: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._runtime_state = new_state
            if new_state == "RUNNING" and self._started_at is None:
                self._started_at = timestamp

    def increment_scheduler_iterations(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._scheduler_iterations += 1

    def record_cycle(self, latency_ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._cycle_count += 1
            self._total_cycle_latency_ms += latency_ms

    def record_dispatch(self, latency_ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._dispatch_count += 1
            self._total_dispatch_latency_ms += latency_ms

    def set_subscriber_count(self, count: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._subscriber_count = count

    def set_queue_depth(self, depth: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._queue_depth = depth

    def record_failed_event(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._failed_events += 1

    def record_failed_handler(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._failed_handlers += 1

    def record_runtime_error(self, error_msg: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._runtime_errors.append(error_msg)

    @property
    def runtime_state(self) -> str:
        with self._lock:
            return self._runtime_state

    @property
    def started_at(self) -> Optional[str]:
        with self._lock:
            return self._started_at

    @property
    def scheduler_iterations(self) -> int:
        with self._lock:
            return self._scheduler_iterations

    @property
    def cycle_count(self) -> int:
        with self._lock:
            return self._cycle_count

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._queue_depth

    @property
    def failed_events(self) -> int:
        with self._lock:
            return self._failed_events

    @property
    def failed_handlers(self) -> int:
        with self._lock:
            return self._failed_handlers

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return self._subscriber_count

    @property
    def average_cycle_latency_ms(self) -> float:
        with self._lock:
            if self._cycle_count == 0:
                return 0.0
            return self._total_cycle_latency_ms / self._cycle_count

    @property
    def average_dispatch_latency_ms(self) -> float:
        with self._lock:
            if self._dispatch_count == 0:
                return 0.0
            return self._total_dispatch_latency_ms / self._dispatch_count

    @property
    def runtime_errors(self) -> List[str]:
        with self._lock:
            return list(self._runtime_errors)
