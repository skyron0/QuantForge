from dataclasses import dataclass


@dataclass
class Decision:

    action: str

    confidence: float

    reason: str