import asyncio

from backend.market.collector import MarketCollector
from backend.market.consumer import MarketConsumer


async def main():

    collector = MarketCollector()
    consumer = MarketConsumer()

    await asyncio.gather(
        collector.run(),
        consumer.run()
    )


if __name__ == "__main__":
    asyncio.run(main())