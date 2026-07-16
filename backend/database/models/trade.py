from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import DateTime

from backend.database.base import Base


class Trade(Base):

    __tablename__ = "trades"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    symbol = Column(
        String,
        nullable=False,
    )

    side = Column(
        String,
        nullable=False,
    )

    quantity = Column(
        Float,
        nullable=False,
    )

    entry_price = Column(
        Float,
        nullable=False,
    )

    exit_price = Column(
        Float,
        nullable=True,
    )

    stop_loss = Column(
        Float,
        nullable=True,
    )

    take_profit = Column(
        Float,
        nullable=True,
    )

    pnl = Column(
        Float,
        nullable=True,
    )

    commission = Column(
        Float,
        default=0.0,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    strategy = Column(
        String,
        nullable=True,
    )

    status = Column(
        String,
        nullable=False,
    )

    open_time = Column(
        DateTime,
        nullable=False,
    )

    close_time = Column(
        DateTime,
        nullable=True,
    )