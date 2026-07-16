from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.monitor.state import dashboard_state


class Dashboard:

    def __init__(self):

        self.live = None

    def build(self):

        table = Table.grid(padding=1)

        table.add_row(
            "Exchange",
            "🟢 Connected"
            if dashboard_state.exchange_connected
            else "🔴 Disconnected"
        )

        table.add_row(
            "Last Tick",
            str(dashboard_state.last_tick)
        )

        table.add_row(
            "Last Candle",
            str(dashboard_state.last_candle_time)
        )

        table.add_row(
            "Candles",
            str(dashboard_state.candle_count)
        )

        table.add_section()

        indicators = dashboard_state.indicators

        table.add_row(
            "RSI",
            str(round(indicators.get("rsi", 0), 2))
        )

        table.add_row(
            "EMA20",
            str(round(indicators.get("ema20", 0), 2))
        )

        table.add_row(
            "ADX",
            str(round(indicators.get("adx", 0), 2))
        )

        table.add_row(
            "Decision",
            dashboard_state.decision
        )

        table.add_row(
            "Confidence",
            str(dashboard_state.confidence)
        )

        table.add_row(
            "Signal",
            dashboard_state.signal
        )

        if dashboard_state.error:

            table.add_section()

            table.add_row(
                "Last Error",
                dashboard_state.error
            )

        return Panel(
            Group(
                Text(
                    "QuantForge Live Dashboard",
                    justify="center",
                    style="bold cyan"
                ),
                table
            )
        )

    def start(self):

        self.live = Live(
            self.build(),
            refresh_per_second=4,
            screen=True,
        )

        self.live.start()

    def refresh(self):

        if self.live:

            self.live.update(
                self.build()
            )

    def stop(self):

        if self.live:

            self.live.stop()