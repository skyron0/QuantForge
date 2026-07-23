"""
Online inference engine with strict model-compatibility gate.

Before any model is invoked, the engine verifies:
  - schema_id
  - schema_version
  - schema fingerprint
  - feature count
  - exact feature ordering

Any mismatch fails closed *before* model invocation.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.models import FeatureSnapshot, PredictionResult
from backend.feature_runtime.exceptions import (
    ModelCompatibilityError,
    InferenceRuntimeError,
    InvalidPredictionError,
    ModelUnavailableError,
)

import math


class FeatureInferenceEngine:
    """
    Wraps an arbitrary prediction callable behind a strict compatibility
    gate.

    The *predict_fn* is a ``Callable[[List[float]], float]`` that takes
    the ordered feature vector and returns a scalar prediction.

    For integration with the full InferenceEngine, callers can supply a
    thin adapter as predict_fn.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        predict_fn: Optional[Callable[[List[float]], float]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._schema = schema
        self._predict_fn = predict_fn
        # Model metadata describes what the model was *trained* on.
        self._model_meta: Dict[str, Any] = model_metadata or {
            "schema_id": schema.schema_id,
            "schema_version": schema.schema_version,
            "fingerprint": schema.fingerprint,
            "feature_names": list(schema.feature_names),
        }

    # ── public ────────────────────────────────────────────────────────────

    def predict(self, snapshot: FeatureSnapshot) -> PredictionResult:
        """
        Run pre-inference compatibility gate, then invoke the model.
        """
        self._check_compatibility(snapshot)

        if self._predict_fn is None:
            raise ModelUnavailableError("No prediction function registered")

        start = time.perf_counter()
        try:
            raw = self._predict_fn(list(snapshot.feature_values))
        except Exception as exc:
            raise InferenceRuntimeError(
                f"Prediction function raised: {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if not isinstance(raw, (int, float)):
            raise InvalidPredictionError(
                f"Prediction must be numeric, got {type(raw).__name__}"
            )
        if math.isnan(raw) or math.isinf(raw):
            raise InvalidPredictionError(f"Invalid prediction value: {raw}")

        return PredictionResult(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            prediction=float(raw),
            model_version=self._model_meta.get("model_version", ""),
            feature_version=snapshot.feature_version,
            latency_ms=elapsed_ms,
        )

    # ── compatibility gate ────────────────────────────────────────────────

    def _check_compatibility(self, snapshot: FeatureSnapshot) -> None:
        meta = self._model_meta

        expected_id = meta.get("schema_id", "")
        if expected_id and expected_id != self._schema.schema_id:
            raise ModelCompatibilityError(
                f"schema_id mismatch: model expects '{expected_id}', "
                f"schema has '{self._schema.schema_id}'"
            )

        expected_ver = meta.get("schema_version", "")
        if expected_ver and expected_ver != self._schema.schema_version:
            raise ModelCompatibilityError(
                f"schema_version mismatch: model expects '{expected_ver}', "
                f"schema has '{self._schema.schema_version}'"
            )

        expected_fp = meta.get("fingerprint", "")
        if expected_fp and expected_fp != snapshot.schema_fingerprint:
            raise ModelCompatibilityError(
                f"fingerprint mismatch: model expects '{expected_fp}', "
                f"snapshot has '{snapshot.schema_fingerprint}'"
            )

        expected_names = meta.get("feature_names")
        if expected_names is not None:
            if len(expected_names) != len(snapshot.feature_names):
                raise ModelCompatibilityError(
                    f"feature count mismatch: model expects {len(expected_names)}, "
                    f"snapshot has {len(snapshot.feature_names)}"
                )
            if list(expected_names) != list(snapshot.feature_names):
                raise ModelCompatibilityError(
                    f"feature ordering mismatch: model expects {expected_names}, "
                    f"snapshot has {list(snapshot.feature_names)}"
                )
