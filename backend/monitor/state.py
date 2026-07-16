from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DashboardState:

    exchange_connected: bool = False

    last_tick: float | None = None

    last_candle_time: datetime | None = None

    candle_count: int = 0

    indicators: dict = field(default_factory=dict)

    decision: str = "-"

    confidence: float = 0.0

    signal: str = "-"

    error: str = ""


dashboard_state = DashboardState()