import numpy as np
import pandas as pd
from typing import List


class DatasetValidationError(Exception):
    """Raised when a dataset fails pre-training validation."""
    pass


class DatasetValidator:
    """Pre-training dataset quality checks."""

    def __init__(self, imbalance_threshold: float = 0.05):
        self.imbalance_threshold = imbalance_threshold

    def validate(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str,
    ) -> None:
        self._check_empty(df)
        self._check_missing_columns(df, feature_columns, label_column)
        self._check_nan(df, feature_columns, label_column)
        self._check_infinite(df, feature_columns)
        self._check_constant_columns(df, feature_columns)
        self._check_single_class(df, label_column)
        self._check_class_imbalance(df, label_column)

    def _check_empty(self, df: pd.DataFrame) -> None:
        if len(df) == 0:
            raise DatasetValidationError("Dataset is empty.")

    def _check_missing_columns(
        self, df: pd.DataFrame, feature_columns: List[str], label_column: str
    ) -> None:
        required = set(feature_columns) | {label_column}
        missing = required - set(df.columns)
        if missing:
            raise DatasetValidationError(
                f"Missing columns: {sorted(missing)}"
            )

    def _check_nan(
        self, df: pd.DataFrame, feature_columns: List[str], label_column: str
    ) -> None:
        cols = feature_columns + [label_column]
        nan_cols = [c for c in cols if df[c].isna().any()]
        if nan_cols:
            raise DatasetValidationError(
                f"NaN values found in columns: {nan_cols}"
            )

    def _check_infinite(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> None:
        for col in feature_columns:
            if df[col].dtype.kind in ("f", "i", "u"):
                if np.isinf(df[col]).any():
                    raise DatasetValidationError(
                        f"Infinite values found in column: {col}"
                    )

    def _check_constant_columns(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> None:
        constant_cols = [c for c in feature_columns if df[c].nunique() <= 1]
        if constant_cols:
            raise DatasetValidationError(
                f"Constant columns (zero variance): {constant_cols}"
            )

    def _check_single_class(
        self, df: pd.DataFrame, label_column: str
    ) -> None:
        if df[label_column].nunique() <= 1:
            raise DatasetValidationError(
                f"Single-class target: only {df[label_column].unique().tolist()} found."
            )

    def _check_class_imbalance(
        self, df: pd.DataFrame, label_column: str
    ) -> None:
        counts = df[label_column].value_counts(normalize=True)
        min_ratio = counts.min()
        if min_ratio < self.imbalance_threshold:
            raise DatasetValidationError(
                f"Severe class imbalance: minority class ratio = {min_ratio:.4f} "
                f"(threshold = {self.imbalance_threshold})"
            )
