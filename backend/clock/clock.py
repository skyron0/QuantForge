from datetime import datetime, timezone


class Clock:

    def __init__(self):
        self._custom_time = None

    def now(self) -> datetime:
        return (
            self._custom_time
            if self._custom_time is not None
            else datetime.now(timezone.utc).replace(tzinfo=None)
        )

    def set_time(self, dt: datetime):
        self._custom_time = dt

    def reset(self):
        self._custom_time = None
