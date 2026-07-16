from configs.logging import app_logger

from backend.indicator.rsi import RSIIndicator
from backend.indicator.ema import EMAIndicator
from backend.indicator.macd import MACDIndicator
from backend.indicator.bollinger import BollingerIndicator
from backend.indicator.atr import ATRIndicator
from backend.indicator.adx import ADX
from backend.indicator.vwap import VWAPIndicator


class IndicatorEngine:

    MIN_CANDLES = 50

    def __init__(self):

        self.rsi = RSIIndicator()
        self.ema = EMAIndicator()
        self.macd = MACDIndicator()
        self.bollinger = BollingerIndicator()
        self.atr = ATRIndicator()
        self.adx = ADX()
        self.vwap = VWAPIndicator()

    def calculate(self, candles):

        if len(candles) < self.MIN_CANDLES:

            app_logger.info(
                f"IndicatorEngine waiting... "
                f"{len(candles)}/{self.MIN_CANDLES} candles"
            )

            return None

        try:

            closes = [float(c.close) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            volumes = [float(c.volume) for c in candles]

            rsi = self.rsi.calculate(closes)
            ema20 = self.ema.calculate(closes, 20)
            macd = self.macd.calculate(closes)
            atr = self.atr.calculate(highs, lows, closes)
            adx = self.adx.calculate(highs, lows, closes)
            vwap = self.vwap.calculate(
                highs,
                lows,
                closes,
                volumes,
            )
            bollinger = self.bollinger.calculate(closes)

            app_logger.info(
                f"Indicators -> "
                f"RSI={rsi} | "
                f"EMA20={ema20} | "
                f"MACD={macd} | "
                f"ATR={atr} | "
                f"ADX={adx} | "
                f"VWAP={vwap} | "
                f"BOLL={bollinger}"
            )

            if rsi is None:
                app_logger.warning("RSI returned None")
                return None

            if ema20 is None:
                app_logger.warning("EMA20 returned None")
                return None

            if macd is None:
                app_logger.warning("MACD returned None")
                return None

            if atr is None:
                app_logger.warning("ATR returned None")
                return None

            if adx is None:
                app_logger.warning("ADX returned None")
                return None

            if vwap is None:
                app_logger.warning("VWAP returned None")
                return None

            if bollinger is None:
                app_logger.warning("Bollinger returned None")
                return None

            return {
                "rsi": rsi,
                "ema20": ema20,

                "macd": macd["macd"],
                "macd_signal": macd["signal"],
                "macd_histogram": macd["histogram"],

                "atr": atr,
                "adx": adx,
                "vwap": vwap,

                "bb_upper": bollinger["upper"],
                "bb_middle": bollinger["middle"],
                "bb_lower": bollinger["lower"],
            }

        except Exception:
            app_logger.exception("IndicatorEngine failed")
            return None