import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional


class TradingSession:
    """
    Mutable, thread-safe session tracker for runtime lifecycle execution.
    Generates a unique session_id, tracks monotonic trading cycles,
    keeps start/last active timestamps, and preserves session-level metadata.
    """
    def __init__(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self._lock = threading.Lock()
        self._session_id = session_id or str(uuid.uuid4())
        self._cycle_counter = 0
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_active_at = self._started_at
        self._metadata = dict(metadata or {})

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def started_at(self) -> str:
        return self._started_at

    @property
    def last_active_at(self) -> str:
        with self._lock:
            return self._last_active_at

    @property
    def cycle_counter(self) -> int:
        with self._lock:
            return self._cycle_counter

    @property
    def metadata(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metadata)

    def increment_cycle(self) -> int:
        """Increments and returns the updated cycle counter."""
        with self._lock:
            self._cycle_counter += 1
            self._last_active_at = datetime.now(timezone.utc).isoformat()
            return self._cycle_counter

    def update_metadata(self, key: str, value: Any) -> None:
        with self._lock:
            self._metadata[key] = value
            self._last_active_at = datetime.now(timezone.utc).isoformat()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._metadata.get(key, default)
