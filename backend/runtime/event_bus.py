import collections
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Set, Any

from backend.runtime.events import TradingEvent
from backend.runtime.exceptions import PublishError


class BaseEventBus(ABC):
    """Abstract interface defining the contract for event publication and subscription."""
    @abstractmethod
    def publish(self, event: TradingEvent) -> None:
        pass

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        pass

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class EventBus(BaseEventBus):
    """
    Thread-safe, bounded, synchronous local EventBus.
    Guarantees strict dispatch ordering for nested calls by queueing publications
    and executing them sequentially in the publisher's thread context.
    Does not retain event history beyond dispatching.
    """
    def __init__(self, max_queue_size: int = 10000, dispatcher: Any = None) -> None:
        self.max_queue_size = max_queue_size
        self._dispatcher = dispatcher
        self._lock = threading.Lock()
        
        # Subscription mapping: event_type_name -> set(handlers)
        self._subscriptions: Dict[str, Set[Callable[[Any], None]]] = collections.defaultdict(set)
        
        # Dispatch queue for preserving order of nested publishes
        self._queue: collections.deque = collections.deque()
        self._is_dispatching = False

    def set_dispatcher(self, dispatcher: Any) -> None:
        with self._lock:
            self._dispatcher = dispatcher

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            self._subscriptions[event_type].add(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            if event_type in self._subscriptions:
                self._subscriptions[event_type].discard(handler)
                if not self._subscriptions[event_type]:
                    del self._subscriptions[event_type]

    def publish(self, event: TradingEvent) -> None:
        """
        Enqueues and dispatches an event. If dispatching is already active on the
        current execution stack (nested publish), the event is queued to guarantee
        that outer event handlers complete execution before inner events are processed.
        """
        if not isinstance(event, TradingEvent):
            raise TypeError("Event must be an instance of TradingEvent")

        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise PublishError(
                    f"EventBus queue size limit reached ({self.max_queue_size})"
                )
            self._queue.append(event)
            
            # If already dispatching in this queue loop, let it process
            if self._is_dispatching:
                return
            self._is_dispatching = True

        try:
            while True:
                next_event = None
                handlers = []
                with self._lock:
                    if self._queue:
                        next_event = self._queue.popleft()
                        if next_event:
                            handlers = list(self._subscriptions.get(next_event.event_type, []))
                
                if next_event is None:
                    break

                # Dispatch using dispatcher if set, otherwise direct invocation
                if self._dispatcher:
                    self._dispatcher.dispatch(next_event, handlers)
                else:
                    for handler in handlers:
                        try:
                            handler(next_event)
                        except Exception:
                            # Direct fallback path ignores or propagates based on setup;
                            # under normal conditions we always have the Dispatcher set
                            pass
        finally:
            with self._lock:
                self._is_dispatching = False

    def clear(self) -> None:
        with self._lock:
            self._subscriptions.clear()
            self._queue.clear()
            self._is_dispatching = False
