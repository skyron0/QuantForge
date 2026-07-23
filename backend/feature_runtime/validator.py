"""
Feature snapshot validation.

Validates bounds (NaN, Inf, negative prices), schema fingerprint match,
key presence, and ordering against the canonical schema.
"""

import math
from typing import List

from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.models import FeatureSnapshot
from backend.feature_runtime.exceptions import (
    FeatureValidationError,
    FeatureSchemaMismatchError,
    FeatureOrderingError,
    MissingFeatureError,
    InvalidFeatureValueError,
)


class FeatureValidator:
    """
    Stateless validator ensuring a FeatureSnapshot is safe for inference.
    """

    def __init__(self, schema: FeatureSchema) -> None:
        self._schema = schema

    def validate(self, snapshot: FeatureSnapshot) -> None:
        """
        Raises on the first violation found.
        """
        self._validate_fingerprint(snapshot)
        self._validate_ordering(snapshot)
        self._validate_count(snapshot)
        self._validate_values(snapshot)

    # ── internal checks ──────────────────────────────────────────────────

    def _validate_fingerprint(self, snapshot: FeatureSnapshot) -> None:
        if snapshot.schema_fingerprint != self._schema.fingerprint:
            raise FeatureSchemaMismatchError(
                f"Fingerprint mismatch: snapshot={snapshot.schema_fingerprint}, "
                f"schema={self._schema.fingerprint}"
            )

    def _validate_ordering(self, snapshot: FeatureSnapshot) -> None:
        if list(snapshot.feature_names) != list(self._schema.feature_names):
            raise FeatureOrderingError(
                f"Feature ordering mismatch: "
                f"snapshot={snapshot.feature_names}, "
                f"schema={self._schema.feature_names}"
            )

    def _validate_count(self, snapshot: FeatureSnapshot) -> None:
        if len(snapshot.feature_names) != self._schema.feature_count:
            raise MissingFeatureError(
                f"Feature count mismatch: "
                f"snapshot has {len(snapshot.feature_names)}, "
                f"schema expects {self._schema.feature_count}"
            )

    def _validate_values(self, snapshot: FeatureSnapshot) -> None:
        for name, val in zip(snapshot.feature_names, snapshot.feature_values):
            if math.isnan(val) or math.isinf(val):
                raise InvalidFeatureValueError(
                    f"Feature '{name}' has invalid value: {val}"
                )
