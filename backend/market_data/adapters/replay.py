from typing import Any, Dict, List, Optional, Set
from backend.market_data.provider import BaseMarketDataProvider
from backend.market_data.models import MarketDataType
from backend.market_data.service import MarketDataService


class ReplayMarketDataProvider(BaseMarketDataProvider):
    """
    Historical replay driver supporting manual stepped ingestion of pre-loaded raw lists.
    """

    def __init__(self, service: Optional[MarketDataService] = None) -> None:
        self.service = service
        self._running = False
        self._subscribed: Set[str] = set()
        self._queue: List[Dict[str, Any]] = []
        self._health_status = "DISCONNECTED"
        self._cursor = 0

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

    def load_data(self, data_list: List[Dict[str, Any]]) -> None:
        self._queue = list(data_list)
        self._cursor = 0

    def step(self) -> Optional[Any]:
        """
        Feeds the next available payload from the queue into MarketDataService and returns envelope.
        """
        if not self._running:
            return None
        if self._cursor >= len(self._queue):
            return None
        if self.service is None:
            raise RuntimeError("ReplayMarketDataProvider service must be initialized before stepping")

        msg = self._queue[self._cursor]
        raw_sym = msg.get("symbol", "")
        # Normalizer check
        normalized_sym = self.service.normalizer.normalize_symbol("replay", raw_sym)
        if normalized_sym not in self._subscribed:
            # Advance cursor but do not ingest if symbol is unsubscribed
            self._cursor += 1
            return None

        self._cursor += 1
        data_type = MarketDataType(msg["data_type"])
        return self.service.ingest_raw_message(
            provider="replay",
            data_type=data_type,
            raw_payload=msg["payload"]
        )

    def has_next(self) -> bool:
        return self._cursor < len(self._queue)
