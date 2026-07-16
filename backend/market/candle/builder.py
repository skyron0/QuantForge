from backend.market.candle.models import Candle


class CandleBuilder:

    def __init__(self, symbol):

        self.symbol = symbol
        self.current = None

    def update(self, tick):

        minute = tick.timestamp.replace(second=0, microsecond=0)
        
        print(f"RAW TIMESTAMP: {tick.timestamp}")
        
        print(
    f"[BUILDER] Tick minute={minute} | Current={self.current.open_time if self.current else None}"
)

        # İlk candle
        if self.current is None:

            self.current = Candle(
                symbol=self.symbol,
                timeframe="1m",
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                open_time=minute,
            )

            return None

        # Yeni dakika başladıysa
        if minute != self.current.open_time:
            
            print("[BUILDER] >>> CANDLE CLOSED <<<")

            finished = self.current

            self.current = Candle(
                symbol=self.symbol,
                timeframe="1m",
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume,
                open_time=minute,
            )

            return finished

        # Aynı dakika devam ediyor

        self.current.close = tick.price

        self.current.high = max(
            self.current.high,
            tick.price
        )

        self.current.low = min(
            self.current.low,
            tick.price
        )

        self.current.volume += tick.volume

        return None