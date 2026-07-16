from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:

    symbol: str

    side: str

    quantity: float

    entry_price: float

    exit_price: float

    pnl: float

    open_time: datetime

    close_time: datetime

    reason: str