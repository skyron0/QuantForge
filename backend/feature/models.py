from dataclasses import dataclass


@dataclass
class FeatureSet:

    symbol: str

    rsi: float

    ema20: float

    macd: float
    macd_signal: float
    macd_histogram: float

    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float

    atr: float

    adx: float

    vwap: float
    
    close: float