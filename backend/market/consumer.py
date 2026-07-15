from configs.logging import app_logger

from backend.database.session import SessionLocal
from backend.market.queue.market_queue import market_queue

from backend.repositories.market_repository import MarketRepository
from backend.repositories.candle_repository import CandleRepository

from backend.market.candle.aggregator import CandleAggregator
from backend.feature.feature_engine import FeatureEngine


class MarketConsumer:

    async def run(self):

        db = SessionLocal()

        repository = MarketRepository(db)
        candle_repository = CandleRepository(db)

        aggregator = CandleAggregator()
        feature_engine = FeatureEngine()

        try:

            while True:

                tick = await market_queue.get()

                # Tick'i kaydet
                repository.save_tick(tick)

                # Candle oluştur
                candle = aggregator.process_tick(tick)

                # Candle tamamlandıysa
                if candle:

                    candle_repository.save(candle)

                    app_logger.info(
                        f"Candle Closed -> "
                        f"{candle.symbol} "
                        f"O:{candle.open} "
                        f"H:{candle.high} "
                        f"L:{candle.low} "
                        f"C:{candle.close}"
                    )

                    candles = candle_repository.get_last(
                        candle.symbol,
                        limit=200
                    )

                    features = feature_engine.build(candles)

                    if features:

                        app_logger.info(
                            f"Features -> "
                            f"RSI:{features.rsi:.2f} | "
                            f"EMA20:{features.ema20:.2f} | "
                            f"MACD:{features.macd:.4f} | "
                            f"ADX:{features.adx:.2f}"
                        )

                app_logger.info(
                    f"Saved -> {tick.symbol} {tick.price}"
                )

                market_queue.task_done()

        finally:

            db.close()