from backend.database.base import Base
from backend.database.engine import engine

# Modelleri import et
import backend.database.models


def init_database():

    Base.metadata.create_all(bind=engine)