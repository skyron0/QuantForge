from configs.logging import app_logger

from backend.market.queue.market_queue import market_queue


class MarketConsumer:

    async def run(self):

        while True:

            tick = await market_queue.get()

            app_logger.info(
                f"Consumed -> {tick.symbol} {tick.price}"
            )

            market_queue.task_done()