import math
from typing import Dict, Any, List, Optional
from backend.inference.exceptions import SchemaValidationError
from backend.inference.models import InferenceRequest


def validate_features(
    features: Dict[str, float],
    model_feature_columns: List[str],
    request_feature_version: Optional[str] = None,
    expected_feature_version: str = "v1",
) -> List[float]:
    """
    Validates features dict against the canonical model training schema.
    Returns the features aligned as a list in the exact canonical order.
    Fails closed on any schema incompatibility by raising SchemaValidationError.
    """
    # 1. Feature version check
    if request_feature_version is not None and request_feature_version != expected_feature_version:
        raise SchemaValidationError(
            f"Feature version mismatch. Expected '{expected_feature_version}', got '{request_feature_version}'."
        )

    # 2. Check for missing features
    missing_features = [col for col in model_feature_columns if col not in features]
    if missing_features:
        raise SchemaValidationError(
            f"Missing required feature columns: {missing_features}"
        )

    # 3. Check for unexpected features
    unexpected_features = [col for col in features if col not in model_feature_columns]
    if unexpected_features:
        raise SchemaValidationError(
            f"Unexpected feature columns in request: {unexpected_features}"
        )

    # 4. Dimensionality validation
    if len(features) != len(model_feature_columns):
        raise SchemaValidationError(
            f"Dimensionality mismatch. Expected {len(model_feature_columns)} features, got {len(features)}."
        )

    # 5. Type and numeric sanity checks (NaN, Inf) and canonical ordering alignment
    ordered_features: List[float] = []
    for col in model_feature_columns:
        val = features[col]
        if val is None:
            raise SchemaValidationError(
                f"Feature '{col}' has null/None value."
            )

        # Basic type representation verification
        try:
            f_val = float(val)
        except (ValueError, TypeError):
            raise SchemaValidationError(
                f"Feature '{col}' has invalid non-numeric value: {val} (type={type(val).__name__})."
            )

        if math.isnan(f_val):
            raise SchemaValidationError(
                f"Feature '{col}' is NaN (invalid numeric value)."
            )
        if math.isinf(f_val):
            raise SchemaValidationError(
                f"Feature '{col}' is infinite (inf) (invalid numeric value)."
            )

        ordered_features.append(f_val)

    return ordered_features
