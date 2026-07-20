from datetime import datetime

from sqlalchemy.orm import Session

from backend.models.candle import Candle


class CandleRepository:

    def __init__(self, db: Session, clock=None):

        self.db = db
        self.model = Candle
        self.clock = clock

    def save(self, candle):

        row = self.model(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            open_time=candle.open_time,
        )

        self.db.add(row)
        self.db.commit()

    def get_last(self, symbol: str, limit: int = 200):

        query = self.db.query(self.model).filter(self.model.symbol == symbol)
        if self.clock is not None:
            query = query.filter(self.model.open_time <= self.clock.now())

        rows = (
            query.order_by(self.model.open_time.desc())
            .limit(limit)
            .all()
        )

        rows.reverse()

        return rows

    def get_between(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ):

        return (
            self.db.query(self.model)
            .filter(
                self.model.symbol == symbol,
                self.model.open_time >= start,
                self.model.open_time <= end,
            )
            .order_by(self.model.open_time.asc())
            .all()
        )