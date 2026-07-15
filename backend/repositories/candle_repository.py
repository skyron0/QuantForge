from sqlalchemy.orm import Session

from backend.models.candle import Candle


class CandleRepository:

    def __init__(self, db: Session):

        self.db = db
        self.model = Candle

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

        rows = (
            self.db.query(self.model)
            .filter(self.model.symbol == symbol)
            .order_by(self.model.open_time.desc())
            .limit(limit)
            .all()
        )

        rows.reverse()

        return rows