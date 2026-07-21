import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List
import uuid

from backend.decision.models import IntelligenceSnapshot
from backend.decision.exceptions import ContextStoreError
from backend.intelligence.models import ReasoningResult


class IntelligenceContextStore:
    """
    Thread-safe, TTL-controlled memory store for IntelligenceSnapshots.
    Keeps track of historical contextual reasoning and controls freshness.
    """

    def __init__(self, default_ttl_seconds: float = 300.0, max_size: int = 1000):
        self._default_ttl_seconds = default_ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        # Key: (symbol, timeframe), Value: IntelligenceSnapshot
        self._snapshots: Dict[Tuple[str, str], IntelligenceSnapshot] = {}
        # Order tracker for bounded memory (FIFO eviction)
        self._insertion_order: List[Tuple[str, str]] = []

    def put(
        self,
        symbol: str,
        timeframe: str,
        result: ReasoningResult,
        ttl_seconds: Optional[float] = None,
    ) -> IntelligenceSnapshot:
        """
        Transforms a ReasoningResult into an IntelligenceSnapshot and stores it thread-safely.
        Prevents unbounded memory growth by evicting older snapshots if limit is reached.
        """
        if not symbol or not timeframe:
            raise ContextStoreError("Symbol and timeframe must be non-empty strings")

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds

        # Parse timestamp or fallback
        try:
            gen_time = datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
        except Exception:
            gen_time = datetime.now(timezone.utc)

        expires_time = gen_time + timedelta(seconds=ttl)

        snapshot = IntelligenceSnapshot(
            snapshot_id=str(uuid.uuid4()),
            symbol=symbol,
            timeframe=timeframe,
            market_regime=result.market_regime,
            directional_bias=result.directional_bias,
            confidence=result.confidence,
            risk_flags=list(result.risk_flags),
            evidence=list(result.evidence),
            reasoning_summary=result.reasoning_summary,
            provider=result.provider,
            model=result.model,
            request_id=result.request_id,
            prompt_id=result.prompt_id,
            prompt_version=result.prompt_version,
            generated_at=gen_time.isoformat(),
            expires_at=expires_time.isoformat(),
            latency_ms=result.latency_ms,
        )

        key = (symbol, timeframe)

        with self._lock:
            if key in self._snapshots:
                # Update existing and move key to end of insertion order for FIFO lifecycle
                self._snapshots[key] = snapshot
                if key in self._insertion_order:
                    self._insertion_order.remove(key)
                self._insertion_order.append(key)
            else:
                # Evict oldest if max size reached
                if len(self._snapshots) >= self._max_size:
                    if self._insertion_order:
                        oldest_key = self._insertion_order.pop(0)
                        self._snapshots.pop(oldest_key, None)
                self._snapshots[key] = snapshot
                self._insertion_order.append(key)

        return snapshot

    def get(
        self, symbol: str, timeframe: str, now: Optional[datetime] = None
    ) -> Optional[IntelligenceSnapshot]:
        """
        Retrieves snapshot. Returns None if missing or expired.
        """
        key = (symbol, timeframe)
        check_time = now if now is not None else datetime.now(timezone.utc)

        # Ensure timezone-aware comparison
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)

        with self._lock:
            snapshot = self._snapshots.get(key)

        if not snapshot:
            return None

        try:
            exp_time = datetime.fromisoformat(snapshot.expires_at)
            # Ensure timezone-aware
            if exp_time.tzinfo is None:
                exp_time = exp_time.replace(tzinfo=timezone.utc)
        except Exception:
            # If parse fails, treat as expired
            return None

        if check_time > exp_time:
            # Clean up expired snapshot from store
            with self._lock:
                self._snapshots.pop(key, None)
                if key in self._insertion_order:
                    self._insertion_order.remove(key)
            return None

        return snapshot

    def invalidate(self, symbol: str, timeframe: str) -> None:
        """
        Explicitly invalidates (removes) a snapshot.
        """
        key = (symbol, timeframe)
        with self._lock:
            self._snapshots.pop(key, None)
            if key in self._insertion_order:
                self._insertion_order.remove(key)

    def clear(self) -> None:
        """
        Clears all snapshots from the store.
        """
        with self._lock:
            self._snapshots.clear()
            self._insertion_order.clear()
