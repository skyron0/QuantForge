class PersistenceError(Exception):
    """Base exception for all persistence errors."""
    pass


class PersistenceValidationError(PersistenceError):
    """Raised when data validation fails before or after persistence."""
    pass


class PersistenceConnectionError(PersistenceError):
    """Raised when connecting to the persistence backend fails."""
    pass


class PersistenceWriteError(PersistenceError):
    """Raised when writing to the persistence storage fails."""
    pass


class PersistenceReadError(PersistenceError):
    """Raised when reading from the persistence storage fails."""
    pass


class RecordNotFoundError(PersistenceError):
    """Raised when a requested record is not found in the persistence layer."""
    pass


class PersistenceSerializationError(PersistenceError):
    """Raised when serialization or deserialization fails."""
    pass


class DuplicateRecordError(PersistenceError):
    """Raised when attempting to persist a duplicate record with conflicting data."""
    pass


class RepositoryError(PersistenceError):
    """Raised when repository layer operations fail."""
    pass


class MigrationError(PersistenceError):
    """Raised when schema migrations fail."""
    pass


class AuditIntegrityError(PersistenceError):
    """Raised when audit log tamper checking or chain validation fails."""
    pass


class UnsupportedPersistenceBackendError(PersistenceError):
    """Raised when an invalid backend mode is selected."""
    pass
