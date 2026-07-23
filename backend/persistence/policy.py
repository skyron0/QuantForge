from dataclasses import dataclass
from typing import Optional
from backend.persistence.exceptions import PersistenceValidationError


@dataclass(frozen=True)
class PersistencePolicy:
    """Validates and enforces parameters for the persistence layer."""
    persistence_enabled: bool = True
    persistence_backend: str = "postgres"  # "postgres" or "memory"
    database_url: str = ""
    database_pool_size: int = 5
    database_pool_overflow: int = 10
    database_timeout_seconds: float = 5.0
    hashing_enabled: bool = True

    def __post_init__(self):
        if self.persistence_enabled:
            if self.persistence_backend not in ("postgres", "memory"):
                raise PersistenceValidationError(
                    f"Unsupported persistence backend: {self.persistence_backend}. Must be 'postgres' or 'memory'."
                )
            if self.persistence_backend == "postgres" and not self.database_url:
                raise PersistenceValidationError(
                    "DATABASE_URL must be provided when postgres persistence is enabled."
                )

        if self.database_pool_size <= 0:
            raise PersistenceValidationError(
                f"database_pool_size must be strictly positive, got {self.database_pool_size}"
            )
        if self.database_pool_overflow < 0:
            raise PersistenceValidationError(
                f"database_pool_overflow cannot be negative, got {self.database_pool_overflow}"
            )
        if self.database_timeout_seconds <= 0.0:
            raise PersistenceValidationError(
                f"database_timeout_seconds must be strictly positive, got {self.database_timeout_seconds}"
            )
