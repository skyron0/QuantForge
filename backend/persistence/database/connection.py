from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from configs.settings import settings
from backend.persistence.exceptions import PersistenceConnectionError

# Global engine and sessionmaker instances
_engine = None
_SessionFactory = None


def get_engine():
    """Initializes and returns the database engine dynamically, with configuration parameters."""
    global _engine
    if _engine is None:
        try:
            # PostgreSQL connection args
            connect_args = {}
            if "postgresql" in settings.DATABASE_URL:
                # connect_timeout parameter uses float or int seconds for psycopg2
                connect_args["connect_timeout"] = int(settings.DATABASE_CONNECT_TIMEOUT_SECONDS)

            _engine = create_engine(
                settings.DATABASE_URL,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_POOL_MAX_OVERFLOW,
                connect_args=connect_args
            )
        except Exception as e:
            raise PersistenceConnectionError(f"Failed to create database engine: {str(e)}")
    return _engine


def get_session_factory():
    """Returns the thread-safe session maker factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def db_session():
    """A context manager to yield a DB session, ensuring commit on success, rollback on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


@contextmanager
def begin_transaction(session=None):
    """Context manager for explicit transaction boundary. If an outer session is passed,

    resuses it without committing, delegation of boundaries to outer scope.

    """
    if session is not None:
        yield session
    else:
        with db_session() as s:
            yield s


def get_engine_url() -> str:
    """Returns the current database engine url."""
    return settings.DATABASE_URL
