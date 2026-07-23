"""
Thread-safe, bounded historical feature buffer.

Stores candle records per symbol in arrival order.  The buffer is a pure
storage layer — feature *calculation* is the responsibility of
FeatureExtractor.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass(frozen=True)
class BufferCandle:
    """Minimal candle record held by the buffer."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoricalFeatureBuffer:
    """
    Per-symbol bounded deque of candle observations.

    Thread-safe: multiple writers (market data ingest) may co-exist with
    reader snapshots (feature extraction).
    """

    def __init__(self, capacity: int = 500) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._buffers: Dict[str, Deque[BufferCandle]] = {}

    # ── writes ────────────────────────────────────────────────────────────

    def append(self, symbol: str, candle: BufferCandle) -> None:
        """Append a candle; oldest entries are evicted when capacity is reached."""
        with self._lock:
            if symbol not in self._buffers:
                self._buffers[symbol] = deque(maxlen=self._capacity)
            self._buffers[symbol].append(candle)

    # ── reads ─────────────────────────────────────────────────────────────

    def get_candles(self, symbol: str) -> List[BufferCandle]:
        """Return a snapshot copy of the buffer for *symbol*."""
        with self._lock:
            buf = self._buffers.get(symbol)
            if buf is None:
                return []
            return list(buf)

    def get_candles_up_to(self, symbol: str, timestamp: str) -> List[BufferCandle]:
        """
        Return only candles with timestamp <= *timestamp* (causal filter).

        Comparison is lexicographic on ISO-8601 strings, which preserves
        chronological ordering.
        """
        with self._lock:
            buf = self._buffers.get(symbol)
            if buf is None:
                return []
            return [c for c in buf if c.timestamp <= timestamp]

    def count(self, symbol: str) -> int:
        with self._lock:
            buf = self._buffers.get(symbol)
            return len(buf) if buf else 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                self._buffers.pop(symbol, None)
            else:
                self._buffers.clear()
