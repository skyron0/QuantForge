from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class SystemInfo(Base):

    __tablename__ = "system_info"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(100))

    version: Mapped[str] = mapped_column(String(20))