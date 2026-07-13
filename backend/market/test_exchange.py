import asyncio

from backend.market.exchanges.bybit import BybitExchange


async def main():
    exchange = BybitExchange()

    await exchange.connect()
    await exchange.subscribe("BTCUSDT")

    tick = await exchange.get_tick()

    print(tick)


if __name__ == "__main__":
    asyncio.run(main())