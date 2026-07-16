import asyncio

from configs.logging import app_logger
from configs.settings import settings

from backend.market.exchanges.bybit import BybitExchange
from backend.market.queue.market_queue import market_queue


class MarketCollector:

    def __init__(self):
        self.exchange = BybitExchange()

    async def run(self):

        await self.exchange.connect()

        await self.exchange.subscribe(
            settings.SYMBOLS.split(",")[0]
        )

        while True:

            tick = await self.exchange.get_tick()

            await market_queue.put(tick)

            app_logger.info(
                f"Tick queued -> {tick.symbol} | Price: {tick.price} | Exchange: {tick.exchange}"
            )