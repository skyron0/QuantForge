from abc import ABC, abstractmethod
from typing import Dict, Set


class BaseMarketDataProvider(ABC):
    """
    Abstract base class for all market data providers (live websocket, memory test, replay).
    """

    @abstractmethod
    def start(self) -> None:
        """Starts connection or ingestion thread/loop."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully disconnects and stops data consumption."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Returns True if provider connection is healthy and responsive."""
        pass

    @abstractmethod
    def subscribe(self, symbol: str) -> None:
        """Subscribes to market feed updates for the specified symbol."""
        pass

    @abstractmethod
    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribes from market feed updates for the specified symbol."""
        pass

    @abstractmethod
    def get_subscribed_symbols(self) -> Set[str]:
        """Returns the set of currently subscribed symbols."""
        pass

    @abstractmethod
    def get_health_status(self) -> str:
        """Returns a string description of connection state (e.g. CONNECTED, DISCONNECTED)."""
        pass
