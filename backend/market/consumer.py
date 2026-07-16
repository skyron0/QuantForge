from configs.logging import app_logger

from backend.database.session import SessionLocal
from backend.market.queue.market_queue import market_queue

from backend.repositories.market_repository import MarketRepository
from backend.repositories.candle_repository import CandleRepository

from backend.market.candle.aggregator import CandleAggregator

from backend.indicator.indicator_engine import IndicatorEngine
from backend.feature.feature_engine import FeatureEngine
from backend.decision.decision_engine import DecisionEngine


class MarketConsumer:

    async def run(self):

        db = SessionLocal()

        repository = MarketRepository(db)
        candle_repository = CandleRepository(db)

        aggregator = CandleAggregator()
        indicator_engine = IndicatorEngine()
        feature_engine = FeatureEngine()
        decision_engine = DecisionEngine()

        try:

            while True:

                tick = await market_queue.get()

                repository.save_tick(tick)

                candle = aggregator.process_tick(tick)

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

                    indicators = indicator_engine.calculate(candles)

                    if indicators:

                        features = feature_engine.build(
                            candle,
                            indicators
                        )

                        app_logger.info(
                            f"Features -> "
                            f"RSI:{features.rsi:.2f} | "
                            f"EMA20:{features.ema20:.2f} | "
                            f"MACD:{features.macd:.4f} | "
                            f"ADX:{features.adx:.2f} | "
                            f"ATR:{features.atr:.2f} | "
                            f"VWAP:{features.vwap:.2f}"
                        )

                        decision = decision_engine.decide(features)

                        if decision:

                            app_logger.info(
                                f"Decision -> "
                                f"{decision.action} | "
                                f"Confidence:{decision.confidence:.2f} | "
                                f"{decision.reason}"
                            )

                app_logger.info(
                    f"Saved -> {tick.symbol} {tick.price}"
                )

                market_queue.task_done()

        finally:

            db.close()