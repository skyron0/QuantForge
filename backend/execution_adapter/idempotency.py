import threading
import time
from typing import Dict, Optional, Tuple
from backend.execution_adapter.models import ExecutionResult

class ExecutionIdempotencyState:
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"

class ExecutionIdempotencyStore:
    def __init__(self, ttl_seconds: float = 3600.0, max_capacity: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_capacity = max_capacity
        self._lock = threading.Lock()
        # Maps intent_id -> (state, timestamp, Optional[ExecutionResult])
        self._store: Dict[str, Tuple[str, float, Optional[ExecutionResult]]] = {}

    def _cleanup_unlocked(self, now: float):
        # Remove expired keys
        expired_keys = [
            k for k, v in self._store.items()
            if now - v[1] > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._store[k]

    def _enforce_capacity_unlocked(self):
        # Enforce max capacity by removing oldest keys if capacity exceeded
        if len(self._store) > self.max_capacity:
            sorted_keys = sorted(self._store.keys(), key=lambda k: self._store[k][1])
            excess = len(self._store) - self.max_capacity
            for i in range(excess):
                del self._store[sorted_keys[i]]

    def claim(self, intent_id: str) -> bool:
        """
        Atomically attempts to claim execution rights for an intent.
        Returns True if claim succeeded (intent is now in progress), False if already claimed.
        """
        now = time.time()
        with self._lock:
            self._cleanup_unlocked(now)
            if intent_id in self._store:
                state, _, _ = self._store[intent_id]
                if state == ExecutionIdempotencyState.IN_PROGRESS or state == ExecutionIdempotencyState.COMPLETED:
                    return False
            # Not found or was cleared, claim it
            self._store[intent_id] = (ExecutionIdempotencyState.IN_PROGRESS, now, None)
            self._enforce_capacity_unlocked()
            return True

    def complete(self, intent_id: str, result: ExecutionResult):
        """Marks the intent execution as completed with the given result."""
        now = time.time()
        with self._lock:
            self._store[intent_id] = (ExecutionIdempotencyState.COMPLETED, now, result)
            self._enforce_capacity_unlocked()

    def release(self, intent_id: str):
        """Releases the in-progress claim on the intent so it can be retried."""
        with self._lock:
            if intent_id in self._store:
                state, _, _ = self._store[intent_id]
                if state == ExecutionIdempotencyState.IN_PROGRESS:
                    del self._store[intent_id]

    def get_result(self, intent_id: str) -> Optional[ExecutionResult]:
        """Gets the associated execution result if completed."""
        with self._lock:
            if intent_id in self._store:
                state, _, res = self._store[intent_id]
                if state == ExecutionIdempotencyState.COMPLETED:
                    return res
            return None

    def clear(self):
        """Clears all stored states (primarily for testing purposes)."""
        with self._lock:
            self._store.clear()
