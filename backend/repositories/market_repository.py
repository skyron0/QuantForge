from sqlalchemy.orm import Session

from backend.market.models import MarketTick
from backend.models.market_tick import MarketTickORM


class MarketRepository:

    def __init__(self, db: Session):
        self.db = db

    def save_tick(self, tick: MarketTick):

        db_tick = MarketTickORM(
            symbol=tick.symbol,
            exchange=tick.exchange,
            price=tick.price,
            volume=tick.volume,
            timestamp=tick.timestamp,
        )

        self.db.add(db_tick)
        self.db.commit()

        return db_tick