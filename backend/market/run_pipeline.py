import asyncio

from backend.market.collector import MarketCollector
from backend.market.consumer import MarketConsumer

from backend.monitor.dashboard import Dashboard


async def dashboard_loop(dashboard):

    while True:
        dashboard.refresh()
        await asyncio.sleep(0.25)


async def main():

    dashboard = Dashboard()

    dashboard.start()

    collector = MarketCollector()
    consumer = MarketConsumer()

    try:

        await asyncio.gather(
            collector.run(),
            consumer.run(),
            dashboard_loop(dashboard)
        )

    finally:

        dashboard.stop()


if __name__ == "__main__":
    asyncio.run(main())