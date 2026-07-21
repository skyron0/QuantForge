from dataclasses import dataclass, field
from typing import List
from backend.decision.exceptions import InvalidPolicyError


@dataclass(frozen=True)
class FusionPolicy:
    """
    Validation and container for versioned Decision Fusion rules.
    Controls thresholds, weights, and safety gates for combining ML signals
    with contextual LLM intelligence.
    """

    policy_version: str
    ml_weight: float
    intelligence_weight: float
    minimum_ml_confidence: float
    minimum_fusion_confidence: float
    minimum_agreement_score: float
    allow_ml_only: bool
    reject_on_critical_drift: bool
    reject_on_intelligence_risk_flags: List[str] = field(default_factory=list)
    proposal_ttl_seconds: float = 60.0

    def __post_init__(self):
        # 1. Weight validations
        if self.ml_weight < 0.0 or self.intelligence_weight < 0.0:
            raise InvalidPolicyError("Policy weights must be non-negative.")

        total_weight = self.ml_weight + self.intelligence_weight
        if total_weight <= 0.0:
            raise InvalidPolicyError("Sum of policy weights must be greater than zero.")

        # Normalize weights so they sum to exactly 1.0
        object.__setattr__(self, "ml_weight", self.ml_weight / total_weight)
        object.__setattr__(
            self, "intelligence_weight", self.intelligence_weight / total_weight
        )

        # 2. Confidence & threshold validation
        if not (0.0 <= self.minimum_ml_confidence <= 1.0):
            raise InvalidPolicyError(
                "minimum_ml_confidence must be between 0.0 and 1.0."
            )

        if not (0.0 <= self.minimum_fusion_confidence <= 1.0):
            raise InvalidPolicyError(
                "minimum_fusion_confidence must be between 0.0 and 1.0."
            )

        # Agreement score is between -1.0 (complete disagreement) and 1.0 (complete agreement)
        if not (-1.0 <= self.minimum_agreement_score <= 1.0):
            raise InvalidPolicyError(
                "minimum_agreement_score must be between -1.0 and 1.0."
            )

        if self.proposal_ttl_seconds <= 0.0:
            raise InvalidPolicyError("proposal_ttl_seconds must be greater than zero.")
