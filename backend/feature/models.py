from dataclasses import dataclass


@dataclass
class FeatureVector:

    symbol: str

    rsi: float

    ema20: float

    macd: float

    macd_signal: float

    macd_histogram: float

    adx: float

    atr: float

    bb_upper: float

    bb_middle: float

    bb_lower: float

    vwap: float

    close: float