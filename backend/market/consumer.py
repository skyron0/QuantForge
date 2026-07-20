from configs.logging import app_logger

from backend.database.session import SessionLocal
from backend.database.models.feature_snapshot import FeatureSnapshot

from backend.market.queue.market_queue import market_queue

from backend.repositories.market_repository import MarketRepository
from backend.repositories.candle_repository import CandleRepository
from backend.repositories.feature_snapshot_repository import (
    FeatureSnapshotRepository,
)

from backend.market.candle.aggregator import CandleAggregator

from backend.indicator.indicator_engine import IndicatorEngine
from backend.feature.feature_engine import FeatureEngine
from backend.decision.decision_engine import DecisionEngine
from backend.decision.models import Decision
from backend.signal.signal_validator import SignalValidator

from backend.execution.paper_executor import PaperExecutor

from backend.monitor.state import dashboard_state


class MarketConsumer:

    def __init__(self, db_session=None, clock=None, strategy=None):

        self.db = db_session if db_session is not None else SessionLocal()
        self.clock = clock

        self.market_repository = MarketRepository(self.db)
        self.candle_repository = CandleRepository(self.db, clock=self.clock)
        self.snapshot_repository = FeatureSnapshotRepository(self.db)

        self.aggregator = CandleAggregator()

        self.indicator_engine = IndicatorEngine()
        self.feature_engine = FeatureEngine()
        self.decision_engine = DecisionEngine(strategy=strategy)
        self.signal_validator = SignalValidator()

        self.paper_executor = PaperExecutor(db_session=self.db, clock=self.clock)

    async def run(self):

        try:

            while True:

                tick = await market_queue.get()

                try:

                    self.process_tick(tick)

                except Exception as e:

                    dashboard_state.error = str(e)

                    app_logger.exception(
                        "Consumer pipeline failed"
                    )

                finally:

                    market_queue.task_done()

        finally:

            self.close()

    def process_tick(self, tick):

        self.market_repository.save_tick(tick)

        app_logger.info(
            f"Saved -> {tick.symbol} {tick.price}"
        )

        candle = self.aggregator.process_tick(tick)

        if candle:

            self.process_candle(candle)

    def process_candle(self, candle):

        self.candle_repository.save(candle)
        self.feed_candle(candle)

    def feed_candle(self, candle):

        app_logger.info(
            f"Candle Closed -> "
            f"{candle.symbol} "
            f"O:{candle.open} "
            f"H:{candle.high} "
            f"L:{candle.low} "
            f"C:{candle.close}"
        )

        candles = self.candle_repository.get_last(
            candle.symbol,
            limit=200,
        )

        dashboard_state.last_candle_time = candle.open_time
        dashboard_state.candle_count = len(candles)

        indicators = self.indicator_engine.calculate(
            candles
        )

        if not indicators:
            return

        dashboard_state.indicators = indicators

        features = self.feature_engine.build(
            candle,
            indicators,
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

        decision = self.decision_engine.decide(
            features
        )
        if decision is None:
            decision = Decision(action="HOLD", confidence=1.0, reason="No strategy decision")

        dashboard_state.decision = decision.action
        dashboard_state.confidence = decision.confidence

        app_logger.info(
            f"Decision -> "
            f"{decision.action} | "
            f"Confidence:{decision.confidence:.2f} | "
            f"{decision.reason}"
        )

        signal = self.signal_validator.validate(
            decision
        )

        if signal:

            dashboard_state.signal = signal.action

            app_logger.info(
                f"Signal -> "
                f"{signal.action} | "
                f"Confidence:{signal.confidence:.2f} | "
                f"{signal.reason}"
            )

        else:

            dashboard_state.signal = "-"

            app_logger.info(
                "Signal -> NONE"
            )
        snapshot = FeatureSnapshot(

            timestamp=candle.open_time,

            symbol=candle.symbol,

            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,

            rsi=features.rsi,
            ema20=features.ema20,

            macd=features.macd,
            macd_signal=features.macd_signal,
            macd_histogram=features.macd_histogram,

            atr=features.atr,
            adx=features.adx,
            vwap=features.vwap,

            bb_upper=features.bb_upper,
            bb_middle=features.bb_middle,
            bb_lower=features.bb_lower,

            decision=decision.action,
            confidence=decision.confidence,

            signal=signal.action if signal else "NONE",
        )

        self.snapshot_repository.create(snapshot)

        app_logger.info(
            f"[SNAPSHOT] "
            f"{snapshot.symbol} "
            f"{snapshot.timestamp}"
        )

        self.paper_executor.execute(
            signal,
            candle,
        )

    def close(self):

        try:
            self.paper_executor.close()
        finally:
            self.db.close()