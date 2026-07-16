from configs.logging import app_logger

from backend.database.models.trade import Trade


class TradeRepository:

    def __init__(self, db):

        self.db = db

    def create(self, trade: Trade):

        try:

            self.db.add(trade)

            self.db.commit()

            self.db.refresh(trade)

            return trade

        except Exception:

            self.db.rollback()

            app_logger.exception(
                "TradeRepository.create failed"
            )

            raise

    def update(self, trade: Trade):

        try:

            self.db.commit()

            self.db.refresh(trade)

            return trade

        except Exception:

            self.db.rollback()

            app_logger.exception(
                "TradeRepository.update failed"
            )

            raise

    def get_by_id(self, trade_id: int):

        return (

            self.db.query(Trade)

            .filter(Trade.id == trade_id)

            .first()

        )

    def get_open_trade(self, symbol: str):

        return (

            self.db.query(Trade)

            .filter(

                Trade.symbol == symbol,

                Trade.status == "OPEN"

            )

            .order_by(Trade.id.desc())

            .first()

        )

    def get_all(self):

        return (

            self.db.query(Trade)

            .order_by(Trade.id.desc())

            .all()

        )

    def get_last(self, limit=100):

        return (

            self.db.query(Trade)

            .order_by(Trade.id.desc())

            .limit(limit)

            .all()

        )

    def get_winners(self):

        return (

            self.db.query(Trade)

            .filter(Trade.pnl > 0)

            .all()

        )

    def get_losers(self):

        return (

            self.db.query(Trade)

            .filter(Trade.pnl <= 0)

            .all()

        )