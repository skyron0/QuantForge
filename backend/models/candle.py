from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import String

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from backend.database.base import Base


class Candle(Base):

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(primary_key=True)

    symbol: Mapped[str] = mapped_column(String(20))

    timeframe: Mapped[str] = mapped_column(String(10))

    open: Mapped[float] = mapped_column(Float)

    high: Mapped[float] = mapped_column(Float)

    low: Mapped[float] = mapped_column(Float)

    close: Mapped[float] = mapped_column(Float)

    volume: Mapped[float] = mapped_column(Float)

    open_time: Mapped[datetime] = mapped_column(DateTime)