from backend.persistence.database.schema import Base
from backend.persistence.exceptions import MigrationError
from backend.persistence.database.connection import get_engine


def initialize_schema(engine) -> None:
    """Safe, idempotent initialization of all persistence tables. Does not run destructive operations."""
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        raise MigrationError(f"Database schema initialization failed: {str(e)}")


def run_migrations() -> None:
    """Helper function to initialize the PostgreSQL database schema."""
    engine = get_engine()
    initialize_schema(engine)
