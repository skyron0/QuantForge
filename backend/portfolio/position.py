from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:

    symbol: str

    side: str

    entry_price: float

    quantity: float

    open_time: datetime

    stop_loss: float

    take_profit: float

    close_price: float | None = None

    close_time: datetime | None = None

    pnl: float = 0.0

    is_open: bool = True