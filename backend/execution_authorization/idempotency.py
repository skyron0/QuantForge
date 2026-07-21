import threading
import time
from typing import Dict, Any, Optional


class IdempotencyStore:
    """
    Thread-safe, TTL-controlled, bounded memory store for idempotency keys.
    Prevents duplicate executions of the same logical trade.
    """

    def __init__(self, max_keys: int = 10000):
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self.max_keys = max_keys

    def register_if_absent(self, idempotency_key: str, data: Any, ttl_seconds: float) -> bool:
        """
        Atomically register an idempotency key if it is not already present and active.
        Enforces maximum capacity (max_keys) and automatically prunes expired keys.

        Returns:
            True if registration succeeded, False if a duplicate active key exists or store is full.
        """
        now = time.time()
        with self._lock:
            # Clean expired keys first to free up space
            self._prune_expired_no_lock(now)

            if idempotency_key in self._store:
                item = self._store[idempotency_key]
                if item["expires_at"] > now:
                    return False  # Active duplicate key
                else:
                    # Entry is expired, remove it to overwrite
                    del self._store[idempotency_key]

            # Bounded memory guard
            if len(self._store) >= self.max_keys:
                return False

            # Lock the key
            self._store[idempotency_key] = {
                "data": data,
                "expires_at": now + ttl_seconds,
            }
            return True

    def get(self, idempotency_key: str) -> Optional[Any]:
        """Retrieve data for a key if it is still active, otherwise return None."""
        now = time.time()
        with self._lock:
            if idempotency_key in self._store:
                item = self._store[idempotency_key]
                if item["expires_at"] > now:
                    return item["data"]
                else:
                    del self._store[idempotency_key]
            return None

    def invalidate(self, idempotency_key: str) -> None:
        """Remove a key immediately, forcing it to be invalid/inactive (e.g. on rollback)."""
        with self._lock:
            if idempotency_key in self._store:
                del self._store[idempotency_key]

    def clear(self) -> None:
        """Clear all active and expired keys in the store."""
        with self._lock:
            self._store.clear()

    def _prune_expired_no_lock(self, now: float) -> None:
        expired_keys = [k for k, v in self._store.items() if v["expires_at"] <= now]
        for k in expired_keys:
            del self._store[k]

    def __len__(self) -> int:
        with self._lock:
            self._prune_expired_no_lock(time.time())
            return len(self._store)
