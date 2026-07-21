import threading
import time
from typing import Dict, List

class FillIdempotencyStore:
    """
    Thread-safe, bounded, in-memory store for tracking processed fill IDs.
    
    WARNING: Bounded in-memory idempotency provides development/paper
    runtime protection only. It does NOT provide permanent exactly-once processing
    across process restarts or capacity overflows.
    """
    def __init__(self, ttl_seconds: float = 86400.0, max_capacity: int = 20000):
        self.ttl_seconds = ttl_seconds
        self.max_capacity = max_capacity
        self._lock = threading.Lock()
        # Maps fill_id -> ingestion_timestamp
        self._store: Dict[str, float] = {}

    def _cleanup_unlocked(self, now: float):
        # Remove expired keys
        expired_keys = [
            k for k, timestamp in self._store.items()
            if now - timestamp > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._store[k]

    def _enforce_capacity_unlocked(self):
        if len(self._store) > self.max_capacity:
            # Sort keys by timestamp (oldest first)
            sorted_keys = sorted(self._store.keys(), key=lambda k: self._store[k])
            excess = len(self._store) - self.max_capacity
            for i in range(excess):
                del self._store[sorted_keys[i]]

    def is_processed(self, fill_id: str) -> bool:
        """Checks if a fill has already been processed."""
        with self._lock:
            return fill_id in self._store

    def record(self, fill_id: str):
        """Records that a fill has been processed."""
        now = time.time()
        with self._lock:
            self._cleanup_unlocked(now)
            self._store[fill_id] = now
            self._enforce_capacity_unlocked()

    def remove(self, fill_id: str):
        """Removes a fill record (e.g. for rollback on atomic validation failure)."""
        with self._lock:
            if fill_id in self._store:
                del self._store[fill_id]

    def clear(self):
        """Clears all records."""
        with self._lock:
            self._store.clear()
