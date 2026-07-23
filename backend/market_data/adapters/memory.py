from typing import Set
from backend.market_data.provider import BaseMarketDataProvider


class MemoryMarketDataProvider(BaseMarketDataProvider):
    """
    In-memory mock market data provider for programmatically feeding custom ticks.
    """

    def __init__(self) -> None:
        self._running = False
        self._subscribed: Set[str] = set()
        self._health_status = "DISCONNECTED"

    def start(self) -> None:
        self._running = True
        self._health_status = "CONNECTED"

    def stop(self) -> None:
        self._running = False
        self._health_status = "DISCONNECTED"

    def is_healthy(self) -> bool:
        return self._running and (self._health_status in ("CONNECTED", "DEGRADED"))

    def subscribe(self, symbol: str) -> None:
        self._subscribed.add(symbol.upper())

    def unsubscribe(self, symbol: str) -> None:
        self._subscribed.discard(symbol.upper())

    def get_subscribed_symbols(self) -> Set[str]:
        return set(self._subscribed)

    def get_health_status(self) -> str:
        return self._health_status

    def set_health_status(self, status: str) -> None:
        self._health_status = status
