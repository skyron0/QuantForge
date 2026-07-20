from datetime import datetime, timezone

from configs.logging import app_logger
from backend.database.models.trade import Trade as TradeModel
from backend.database.session import SessionLocal
from backend.risk.risk_manager import RiskManager
from backend.repositories.trade_repository import TradeRepository
from configs.settings import settings

from backend.portfolio.position import Position
from backend.portfolio.portfolio import Portfolio


class PaperExecutor:

    TAKE_PROFIT = settings.TAKE_PROFIT
    STOP_LOSS = settings.STOP_LOSS

    def __init__(self, db_session=None, clock=None):
        self.risk_manager = RiskManager()
        self.db = db_session if db_session is not None else SessionLocal()
        self.clock = clock

        self.trade_repository = TradeRepository(
            self.db
        )

        self.portfolio = Portfolio()

    def execute(self, signal, candle):

        self.portfolio.update_positions(candle)

        for position in self.portfolio.get_open_positions():

            app_logger.info(

                f"[PnL] "
                f"{position.symbol} | "
                f"{position.pnl:.2f}"

            )

            close_reason = None

            if position.pnl >= self.TAKE_PROFIT:
                close_reason = "TAKE_PROFIT"

            elif position.pnl <= -self.STOP_LOSS:
                close_reason = "STOP_LOSS"

            if close_reason:

                position.close_price = candle.close
                position.close_time = self.clock.now() if self.clock is not None else datetime.now(timezone.utc).replace(tzinfo=None)

                trade = self.trade_repository.get_open_trade(
                    position.symbol
                )

                if trade:

                    trade.exit_price = position.close_price

                    trade.pnl = position.pnl

                    trade.status = "CLOSED"

                    trade.close_time = position.close_time

                    trade.reason = close_reason

                    self.trade_repository.update(trade)

                self.portfolio.close_position(position)

                app_logger.info(

                    f"[PAPER CLOSE] "
                    f"{position.symbol} | "
                    f"Exit={position.close_price:.2f} | "
                    f"PnL={position.pnl:.2f} | "
                    f"{close_reason}"

                )

        if signal is None:
            return

        if signal.action != "BUY":
            return

        if self.portfolio.has_open_position(candle.symbol):
            return
        if not self.risk_manager.can_open_position(
            len(self.portfolio.get_open_positions())
        ):
            return

        position = Position(

            symbol=candle.symbol,

            side="BUY",

            entry_price=candle.close,

            quantity=self.risk_manager.calculate_position_size(
            entry_price=candle.close,
            stop_loss=candle.close - self.STOP_LOSS,
        ),

            open_time=self.clock.now() if self.clock is not None else datetime.now(timezone.utc).replace(tzinfo=None),

            stop_loss=candle.close - self.STOP_LOSS,

            take_profit=candle.close + self.TAKE_PROFIT,

        )

        self.portfolio.open_position(position)

        trade = TradeModel(

            symbol=position.symbol,

            side=position.side,

            quantity=position.quantity,

            entry_price=position.entry_price,

            exit_price=None,

            stop_loss=position.stop_loss,

            take_profit=position.take_profit,

            pnl=0.0,

            commission=0.0,

            confidence=getattr(signal, "confidence", None),

            strategy="RuleBased_v1",

            reason=getattr(signal, "reason", "BUY"),

            model_version="v1",

            status="OPEN",

            open_time=position.open_time,

            close_time=None,

        )

        self.trade_repository.create(trade)

        app_logger.info(

            f"[PAPER BUY] "
            f"{position.symbol} | "
            f"Entry={position.entry_price:.2f} | "
            f"TP={position.take_profit:.2f} | "
            f"SL={position.stop_loss:.2f}"

        )

    def close(self):

        self.db.close()