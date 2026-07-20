from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class EquityPoint:

    timestamp: datetime

    equity: float

    drawdown_pct: float

    drawdown_abs: float


@dataclass
class BacktestResult:

    total_trades: int

    winning_trades: int

    losing_trades: int

    win_rate: float

    net_profit: float

    gross_profit: float

    gross_loss: float

    average_profit: float

    average_win: float

    average_loss: float

    profit_factor: float

    largest_win: float

    largest_loss: float

    average_trade_duration: float  # in seconds

    maximum_drawdown_pct: float

    maximum_drawdown_abs: float

    equity_curve: List[EquityPoint] = field(default_factory=list)
