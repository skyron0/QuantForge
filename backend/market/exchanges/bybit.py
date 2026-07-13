from backend.market.core.models import MarketTick
from backend.market.exchanges.base import BaseExchange

from datetime import datetime


class BybitExchange(BaseExchange):

    async def connect(self):
        print("Connected to Bybit")

    async def subscribe(self, symbol: str):
        print(f"Subscribed to {symbol}")

    async def get_tick(self):

        return MarketTick(
            symbol="BTCUSDT",
            price=117000.50,
            volume=0.25,
            timestamp=datetime.utcnow(),
            exchange="bybit"
        )