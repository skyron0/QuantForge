"""
Versioned feature schema with deterministic SHA-256 fingerprint.

The fingerprint is computed from the canonical definition including
schema_id, schema_version, and the *ordered* feature names list.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class FeatureSchema:
    """
    Immutable, versioned contract describing the exact features a model
    expects and their deterministic ordering.

    The fingerprint is always derived — never hardcoded — so that any
    change to the schema definition automatically invalidates stale
    consumers.
    """

    schema_id: str
    schema_version: str
    feature_names: List[str]
    fingerprint: str = field(init=False, repr=True)

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("schema_id must be non-empty")
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")
        if not self.feature_names:
            raise ValueError("feature_names must contain at least one entry")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must not contain duplicates")

        canonical = self._canonical_string()
        fp = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        # frozen=True requires object.__setattr__ for post-init assignment
        object.__setattr__(self, "fingerprint", fp)

    # ── helpers ───────────────────────────────────────────────────────────

    def _canonical_string(self) -> str:
        """
        Deterministic canonical representation used as SHA-256 input.

        Format:  schema_id|schema_version|feat1,feat2,...,featN
        """
        names_csv = ",".join(self.feature_names)
        return f"{self.schema_id}|{self.schema_version}|{names_csv}"

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def matches(self, other: "FeatureSchema") -> bool:
        """True when two schemas are structurally identical."""
        return self.fingerprint == other.fingerprint

    def validate_ordering(self, names: List[str]) -> bool:
        """True when *names* exactly matches the canonical ordering."""
        return list(names) == list(self.feature_names)
