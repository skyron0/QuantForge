import datetime
import logging
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftStatus(str, Enum):
    STABLE = "STABLE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNAVAILABLE = "UNAVAILABLE"


class DriftMetric(str, Enum):
    PSI = "PSI"
    KS = "KS"


class DriftThresholds:
    def __init__(self, psi_warning: float = 0.10, psi_critical: float = 0.25):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical


class FeatureDriftResult:
    def __init__(
        self,
        feature_name: str,
        metric: DriftMetric,
        score: Optional[float],
        status: DriftStatus,
        p_value: Optional[float] = None,
        notes: Optional[str] = None,
    ):
        self.feature_name = feature_name
        self.metric = metric
        self.score = score
        self.status = status
        self.p_value = p_value
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "metric": self.metric.value,
            "score": self.score,
            "status": self.status.value,
            "p_value": self.p_value,
            "notes": self.notes,
        }


class DriftReport:
    def __init__(
        self,
        model_version: str,
        feature_results: List[FeatureDriftResult],
        timestamp: Optional[str] = None,
    ):
        self.model_version = model_version
        self.feature_results = feature_results
        self.timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Compute aggregate counters
        self.total_features_evaluated = len(feature_results)
        self.stable_feature_count = sum(1 for r in feature_results if r.status == DriftStatus.STABLE)
        self.warning_feature_count = sum(1 for r in feature_results if r.status == DriftStatus.WARNING)
        self.critical_feature_count = sum(1 for r in feature_results if r.status == DriftStatus.CRITICAL)
        self.unavailable_feature_count = sum(1 for r in feature_results if r.status == DriftStatus.UNAVAILABLE)

        # Overall status is the worst of the statuses
        if self.critical_feature_count > 0:
            self.overall_drift_status = DriftStatus.CRITICAL
        elif self.warning_feature_count > 0:
            self.overall_drift_status = DriftStatus.WARNING
        elif self.stable_feature_count > 0:
            self.overall_drift_status = DriftStatus.STABLE
        else:
            self.overall_drift_status = DriftStatus.UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "total_features_evaluated": self.total_features_evaluated,
            "stable_feature_count": self.stable_feature_count,
            "warning_feature_count": self.warning_feature_count,
            "critical_feature_count": self.critical_feature_count,
            "unavailable_feature_count": self.unavailable_feature_count,
            "overall_drift_status": self.overall_drift_status.value,
            "features": {r.feature_name: r.to_dict() for r in self.feature_results},
        }


class FeatureDriftDetector:
    def __init__(self, thresholds: Optional[DriftThresholds] = None):
        self.thresholds = thresholds or DriftThresholds()

    def calculate_psi(
        self,
        feature_name: str,
        baseline_stats: Dict[str, Any],
        runtime_observations: np.ndarray,
        epsilon: float = 1e-4,
    ) -> FeatureDriftResult:
        """
        Computes the Population Stability Index (PSI) against quantile baseline bounds.
        """
        if len(runtime_observations) == 0:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.PSI,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes="No runtime observations available.",
            )

        # Retrieve versioned binning metadata
        bin_edges = baseline_stats.get("bin_edges")
        expected_proportions = baseline_stats.get("expected_proportions")

        # Fallback/Migration: If new properties are missing but legacy quantiles exist, reconstruct it
        if not bin_edges or not expected_proportions:
            quantiles = baseline_stats.get("quantiles", {})
            if "25" in quantiles and "50" in quantiles and "75" in quantiles:
                q25 = float(quantiles["25"])
                q50 = float(quantiles["50"])
                q75 = float(quantiles["75"])
                bin_edges = [-np.inf, q25, q50, q75, np.inf]
                expected_proportions = [0.25, 0.25, 0.25, 0.25]
            else:
                return FeatureDriftResult(
                    feature_name=feature_name,
                    metric=DriftMetric.PSI,
                    score=None,
                    status=DriftStatus.UNAVAILABLE,
                    notes="Baseline stats missing bin edges or quantiles.",
                )

        # Convert any string representations (e.g. "-inf", "inf") to floats
        try:
            float_edges = [float(x) if isinstance(x, str) else x for x in bin_edges]
            bin_edges = np.array(float_edges)
        except Exception as e:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.PSI,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes=f"Failed to parse bin edges: {str(e)}",
            )

        expected_proportions = np.array(expected_proportions)
        n_bins = len(expected_proportions)

        # Count runtime observations in each bin
        actual_counts = np.zeros(n_bins)
        try:
            indices = np.digitize(runtime_observations, bin_edges[1:-1])
            for idx in indices:
                # clip to ensure safe indexing
                bin_idx = int(np.clip(idx, 0, n_bins - 1))
                actual_counts[bin_idx] += 1
        except Exception as e:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.PSI,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes=f"Error digitizing values into bins: {str(e)}",
            )

        # Apply Laplace epsilon smoothing to handle zero frequency buckets
        total_actual = len(runtime_observations)
        actual_probs = (actual_counts + epsilon) / (total_actual + epsilon * n_bins)
        
        # Calculate PSI
        # psi_i = (Actual_i - Expected_i) * ln(Actual_i / Expected_i)
        try:
            psi_val = float(np.sum((actual_probs - expected_proportions) * np.log(actual_probs / expected_proportions)))
        except Exception as e:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.PSI,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes=f"Calculation error: {str(e)}",
            )

        # Assign status based on thresholds
        if psi_val < self.thresholds.psi_warning:
            status = DriftStatus.STABLE
        elif psi_val < self.thresholds.psi_critical:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.CRITICAL

        return FeatureDriftResult(
            feature_name=feature_name,
            metric=DriftMetric.PSI,
            score=psi_val,
            status=status,
        )

    def calculate_ks(
        self,
        feature_name: str,
        baseline_stats: Dict[str, Any],
        runtime_observations: np.ndarray,
    ) -> FeatureDriftResult:
        """
        Runs Kolmogorov-Smirnov test against baseline reference.
        Degrades to UNAVAILABLE if raw validation data is missing.
        """
        raw_reference = baseline_stats.get("raw_reference")
        if not raw_reference or len(raw_reference) == 0:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.KS,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes="Raw validation reference data unavailable for KS test.",
            )

        if len(runtime_observations) == 0:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.KS,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes="No runtime observations available.",
            )

        try:
            ks_stat, p_val = stats.ks_2samp(raw_reference, runtime_observations)
        except Exception as e:
            return FeatureDriftResult(
                feature_name=feature_name,
                metric=DriftMetric.KS,
                score=None,
                status=DriftStatus.UNAVAILABLE,
                notes=f"Error executing KS test: {str(e)}",
            )

        # Assume standard 5% significance level for drift detection
        # If p-value < 0.05, we reject the null hypothesis -> distribution shifted
        status = DriftStatus.CRITICAL if p_val < 0.05 else DriftStatus.STABLE

        return FeatureDriftResult(
            feature_name=feature_name,
            metric=DriftMetric.KS,
            score=float(ks_stat),
            status=status,
            p_value=float(p_val),
        )

    def generate_report(
        self,
        model_version: str,
        drift_baseline: Dict[str, Any],
        runtime_observations: Dict[str, List[float]],
    ) -> DriftReport:
        """
        Executes PSI (and KS if available) calculations for all features
        and returns an aggregated DriftReport.
        """
        feature_results = []
        for feature_name, baseline_stats in drift_baseline.items():
            observations = runtime_observations.get(feature_name, [])
            obs_array = np.array(observations, dtype=float)
            
            # Running PSI
            psi_res = self.calculate_psi(feature_name, baseline_stats, obs_array)
            feature_results.append(psi_res)

        return DriftReport(model_version=model_version, feature_results=feature_results)
