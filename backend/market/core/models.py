from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketTick:
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    exchange: str