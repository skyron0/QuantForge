import time
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Any

from backend.runtime.events import TradingEvent
from backend.runtime.exceptions import SubscriberError
from backend.runtime.telemetry import RuntimeTelemetry


class BaseDispatcher(ABC):
    """Abstract interface defining the dispatch capability for events."""
    @abstractmethod
    def dispatch(self, event: TradingEvent, handlers: List[Callable[[Any], None]]) -> None:
        pass


class Dispatcher(BaseDispatcher):
    """
    Delivers events to registered subscribers.
    Isolates handler failures to protect runtime crash safety,
    measures delivery metrics, and updates telemetry.
    """
    def __init__(self, telemetry: Optional[RuntimeTelemetry] = None) -> None:
        self.telemetry = telemetry

    def dispatch(self, event: TradingEvent, handlers: List[Callable[[Any], None]]) -> None:
        """
        Invokes all subscriber callbacks for the given event.
        Guarantees that a failing handler does not prevent other handlers
        from receiving the event.
        """
        if not handlers:
            return

        start_time = time.perf_counter()
        failed_any = False

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                failed_any = True
                if self.telemetry:
                    self.telemetry.record_failed_handler()
                    self.telemetry.record_runtime_error(
                        f"Subscriber handler {handler.__name__ if hasattr(handler, '__name__') else str(handler)} "
                        f"failed for event type {event.event_type}: {str(e)}"
                    )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if self.telemetry:
            self.telemetry.record_dispatch(duration_ms)
            if failed_any:
                self.telemetry.record_failed_event()
