from dataclasses import dataclass
from backend.replay.exceptions import ReplayValidationError

@dataclass(frozen=True)
class ReplayPolicy:
    """
    Simulation policy governing dataset limits, safety parameters, and invariant checks.
    """
    policy_version: str = "3.13.0"
    paper_only: bool = True
    max_dataset_rows: int = 100000
    allow_empty_datasets: bool = False
    max_consecutive_errors: int = 5

    def validate(self) -> None:
        """Enforces that simulation runs with execution safety rules."""
        if not self.paper_only:
            raise ReplayValidationError(
                "Execution Safety Violation: Historical replay must run in paper_only mode."
            )
        if self.max_dataset_rows <= 0:
            raise ReplayValidationError("max_dataset_rows must be a positive integer.")
