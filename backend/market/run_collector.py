import asyncio

from backend.market.collector import MarketCollector


async def main():

    collector = MarketCollector()

    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())