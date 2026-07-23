import abc
from datetime import datetime, timezone
from typing import Optional, Union

class Clock(abc.ABC):
    """
    Abstract Base Clock providing a uniform time-fetching interface.
    """
    @abc.abstractmethod
    def now(self) -> datetime:
        """Returns the current aware UTC datetime."""
        pass

class SystemClock(Clock):
    """
    Returns actual system wall-clock time in UTC timezone.
    Used in paper/live trading.
    """
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

class ReplayClock(Clock):
    """
    Simulated clock with manual, forward-only time advancement.
    Used in historical simulation.
    """
    def __init__(self, initial_time: Optional[Union[str, datetime]] = None) -> None:
        if initial_time is None:
            self._current = datetime.now(timezone.utc)
        elif isinstance(initial_time, str):
            self._current = self._parse_iso_utc(initial_time)
        elif isinstance(initial_time, datetime):
            self._current = self._ensure_utc(initial_time)
        else:
            raise TypeError("initial_time must be str, datetime, or None")

    @property
    def current_time(self) -> datetime:
        return self._current

    def set_time(self, timestamp: Union[str, datetime]) -> None:
        """Explicitly sets the clock time to a specific UTC datetime."""
        dt = (
            self._parse_iso_utc(timestamp) if isinstance(timestamp, str)
            else self._ensure_utc(timestamp)
        )
        self._current = dt

    def advance_to(self, timestamp: Union[str, datetime]) -> None:
        """
        Advances the clock to a future time.
        Raises ValueError if the target time is in the past relative to the current time.
        """
        dt = (
            self._parse_iso_utc(timestamp) if isinstance(timestamp, str)
            else self._ensure_utc(timestamp)
        )
        if dt < self._current:
            raise ValueError(
                f"Cannot advance clock backwards. Current: {self._current.isoformat()}, Target: {dt.isoformat()}"
            )
        self._current = dt

    def reset(self, timestamp: Union[str, datetime]) -> None:
        """Resets the clock to a specific timestamp, bypassing direction validations."""
        dt = (
            self._parse_iso_utc(timestamp) if isinstance(timestamp, str)
            else self._ensure_utc(timestamp)
        )
        self._current = dt

    def now(self) -> datetime:
        return self._current

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            raise ValueError("Timezone-naive datetime is not permitted. Must be UTC.")
        # Ensure UTC timezone specifically
        return dt.astimezone(timezone.utc)

    def _parse_iso_utc(self, ts: str) -> datetime:
        if not ts:
            raise ValueError("Timestamp string cannot be empty")
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(ts)
            return self._ensure_utc(dt)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {ts}") from e
