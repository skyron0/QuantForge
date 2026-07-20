from enum import Enum


class ModelStatus(str, Enum):
    TRAINING = "TRAINING"
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    SHADOW = "SHADOW"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


class InvalidStateTransitionError(Exception):
    """Raised when a model lifecycle state transition violates sequencing rules."""
    pass


# Strict forward flow of active lifecycle
FORWARD_FLOW = {
    ModelStatus.TRAINING: ModelStatus.CANDIDATE,
    ModelStatus.CANDIDATE: ModelStatus.VALIDATED,
    ModelStatus.VALIDATED: ModelStatus.SHADOW,
    ModelStatus.SHADOW: ModelStatus.PRODUCTION,
}


def validate_transition(current: ModelStatus, new: ModelStatus) -> None:
    """
    Enforces the lifecycle state sequencing.
    - Active pipeline flows step-by-step forward.
    - Any state (except ARCHIVED) can transition to DEPRECATED.
    - Any state can transition to ARCHIVED.
    - No states can transition out of ARCHIVED.
    """
    if current == new:
        return

    # Terminal state check
    if current == ModelStatus.ARCHIVED:
        raise InvalidStateTransitionError(
            "Cannot transition out of terminal state ARCHIVED."
        )

    # Transition to retirement states is always allowed
    if new in (ModelStatus.DEPRECATED, ModelStatus.ARCHIVED):
        return

    # Normal step-by-step forward flow check
    expected_next = FORWARD_FLOW.get(current)
    if expected_next != new:
        raise InvalidStateTransitionError(
            f"Invalid lifecycle transition: '{current.value}' cannot transition to '{new.value}'. "
            f"Expected next step: '{expected_next.value if expected_next else 'None'}'"
        )
