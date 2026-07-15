from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class MarketTickORM(Base):
    """
    Database model for market ticks.
    """

    __tablename__ = "market_ticks"

    id: Mapped[int] = mapped_column(primary_key=True)

    symbol: Mapped[str] = mapped_column(String(20), index=True)

    exchange: Mapped[str] = mapped_column(String(20), index=True)

    price: Mapped[float] = mapped_column(Float)

    volume: Mapped[float] = mapped_column(Float)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )