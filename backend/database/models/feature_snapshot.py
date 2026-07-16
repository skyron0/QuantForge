from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import DateTime

from backend.database.base import Base


class FeatureSnapshot(Base):

    __tablename__ = "feature_snapshots"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    timestamp = Column(
        DateTime,
        nullable=False,
    )

    symbol = Column(
        String,
        nullable=False,
    )

    open = Column(
        Float,
        nullable=False,
    )

    high = Column(
        Float,
        nullable=False,
    )

    low = Column(
        Float,
        nullable=False,
    )

    close = Column(
        Float,
        nullable=False,
    )

    volume = Column(
        Float,
        nullable=False,
    )

    rsi = Column(
        Float,
        nullable=True,
    )

    ema20 = Column(
        Float,
        nullable=True,
    )

    macd = Column(
        Float,
        nullable=True,
    )

    macd_signal = Column(
        Float,
        nullable=True,
    )

    macd_histogram = Column(
        Float,
        nullable=True,
    )

    atr = Column(
        Float,
        nullable=True,
    )

    adx = Column(
        Float,
        nullable=True,
    )

    vwap = Column(
        Float,
        nullable=True,
    )

    bb_upper = Column(
        Float,
        nullable=True,
    )

    bb_middle = Column(
        Float,
        nullable=True,
    )

    bb_lower = Column(
        Float,
        nullable=True,
    )

    decision = Column(
        String,
        nullable=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    signal = Column(
        String,
        nullable=True,
    )