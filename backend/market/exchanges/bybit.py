import json

import websockets

from datetime import datetime, UTC

from configs.settings import settings

from backend.market.core.models import MarketTick
from backend.market.exchanges.base import BaseExchange


class BybitExchange(BaseExchange):

    def __init__(self):

        self.ws = None

    async def connect(self):

        self.ws = await websockets.connect(
            settings.BYBIT_WS,
            ping_interval=20,
            ping_timeout=20,
        )

        print("Connected to Bybit")

    async def subscribe(self, symbol: str):

        payload = {
            "op": "subscribe",
            "args": [
                f"publicTrade.{symbol}"
            ]
        }

        await self.ws.send(json.dumps(payload))

        print(f"Subscribed to {symbol}")

    async def get_tick(self):

        while True:

            message = await self.ws.recv()

            data = json.loads(message)

            # Ping / pong / subscribe ack
            if "topic" not in data:
                continue

            if data.get("topic") != "publicTrade.BTCUSDT":
                continue

            trades = data.get("data")

            if not trades:
                continue

            trade = trades[0]

            return MarketTick(
                symbol=trade["s"],
                price=float(trade["p"]),
                volume=float(trade["v"]),
                timestamp=datetime.now(UTC),
                exchange="bybit",
            )