import time
import datetime
from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np

from backend.training.registry import ModelRegistry
from backend.inference.exceptions import (
    InferenceError,
    ModelNotFoundError,
    ModelLoadError,
    LifecycleError,
    SchemaValidationError,
    PredictionError,
    RegistryError,
)
from backend.inference.models import InferenceRequest, InferenceResponse, InferenceMetadata
from backend.inference.model_loader import ModelLoaderCache
from backend.inference.validator import validate_features
from backend.inference.telemetry import InferenceTelemetrySink, LoggerTelemetrySink
from backend.inference.drift import DriftObserver


class InferenceEngine:
    """
    Central orchestration engine for executing real-time machine learning predictions.
    Validates schemas causal-safely, caches models thread-safely, and outputs structured predictions.
    Runs in absolute isolation from trading execution layers.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        loader_cache: Optional[ModelLoaderCache] = None,
        telemetry_sink: Optional[InferenceTelemetrySink] = None,
        drift_observers: Optional[List[DriftObserver]] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.cache = loader_cache or ModelLoaderCache(self.registry)
        self.telemetry_sink = telemetry_sink or LoggerTelemetrySink()
        self.drift_observers = drift_observers or []

    def register_drift_observer(self, observer: DriftObserver) -> None:
        """Appends a feature drift observer state change hook."""
        self.drift_observers.append(observer)

    def predict_one(self, request: InferenceRequest) -> InferenceResponse:
        """
        Executes inference for a single request.
        Returns InferenceResponse on success. On any failure, raises InferenceError subtype.
        Logs telemetric outcomes (both success and failures) before errors propagate.
        """
        t_start = time.perf_counter()
        try:
            # 1. Load model & enforce registry lifecycle authorization (live check)
            model, cache_hit = self.cache.get_model(request.model_version)

            model_record = self.registry.repo.get(request.model_version)
            if not model_record:
                raise ModelNotFoundError(f"Model version '{request.model_version}' not found.")

            # 2. Enforce strict feature schema validation
            aligned_list = validate_features(
                features=request.features,
                model_feature_columns=model.feature_columns,
                request_feature_version=request.feature_version,
                expected_feature_version=model_record.feature_version,
            )

            # 3. Model prediction execution (via monotonic high-res timings)
            X = np.array([aligned_list])
            try:
                pred_slice = model.predict_batch(X)
                prediction_val = float(pred_slice[0])
            except Exception as e:
                raise PredictionError(f"Prediction execution failed: {str(e)}") from e

            # 4. Standard prediction normalization and confidence calibration
            probabilities = None
            confidence = None
            confidence_source = "UNAVAILABLE"
            is_calibrated = False

            if model.task_type == "classification":
                # Fallback check for predict_proba
                proba_slice = None
                if hasattr(model, "predict_proba_batch"):
                    try:
                        proba_slice = model.predict_proba_batch(X)  # type: ignore
                    except Exception:
                        pass
                
                if proba_slice is None and hasattr(model.raw_model, "predict_proba"):
                    try:
                        proba_slice = model.raw_model.predict_proba(X)
                    except Exception:
                        pass

                if proba_slice is not None and len(proba_slice) > 0:
                    calibrator = getattr(model, "calibrator", None)
                    if calibrator is not None:
                        try:
                            # Apply probability calibration
                            calibrated_probs = calibrator.transform(proba_slice)
                            probabilities = calibrated_probs[0].tolist()
                            confidence = float(np.max(calibrated_probs[0]))
                            calib_meta = getattr(model_record, "calibration_metadata", {})
                            calib_method = calib_meta.get("calibration_method", "calibrated")
                            confidence_source = calib_method.upper()
                            is_calibrated = True
                        except Exception as e:
                            # Fail-closed behavior on transform failure
                            raise PredictionError(f"Probability calibration execution failed: {str(e)}") from e
                    else:
                        probabilities = proba_slice[0].tolist()
                        confidence = float(np.max(proba_slice[0]))
                        confidence_source = "UNCALIBRATED"

            # 5. Emit features to drift monitoring hooks
            for observer in self.drift_observers:
                try:
                    observer.on_observation(request.model_version, request.features)
                except Exception as de:
                    # Drift hooks must be isolated so failures do not crash inference
                    pass

            t_end = time.perf_counter()
            latency = (t_end - t_start) * 1000.0

            response = InferenceResponse(
                model_version=request.model_version,
                symbol=request.symbol,
                prediction=prediction_val,
                probabilities=probabilities,
                confidence=confidence,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                latency_ms=latency,
                feature_version=model_record.feature_version,
                metadata=InferenceMetadata(
                    cache_hit=cache_hit,
                    is_calibrated=is_calibrated,
                    confidence_source=confidence_source,
                ),
            )

            # Record success telemetry
            self.telemetry_sink.record_success(request, response)
            return response

        except Exception as error:
            t_end = time.perf_counter()
            latency = (t_end - t_start) * 1000.0

            # Wrap generic Exceptions in PredictionError to retain fail-closed behavior
            if not isinstance(error, InferenceError):
                error = PredictionError(f"Prediction pipeline failed: {str(error)}")

            # Telemetry must still record failures before exception is propagated
            self.telemetry_sink.record_failure(
                request=request,
                error_type=type(error).__name__,
                error_message=str(error),
                latency_ms=latency,
            )
            raise error

    def predict_batch(
        self, requests: List[InferenceRequest]
    ) -> List[Union[InferenceResponse, InferenceError]]:
        """
        Executes optimized batch inference.
        Returns a list of length len(requests), where results[i] is either the
        successful InferenceResponse or the corresponding InferenceError subclass.
        Ensures partial failure isolation and provides independent traceability.
        """
        results: List[Union[InferenceResponse, InferenceError]] = [
            PredictionError("Inference did not run.") for _ in range(len(requests))
        ]

        if not requests:
            return results

        # Group indices by model version to avoid redundant lookups and file loads
        version_groups: Dict[str, List[int]] = {}
        for idx, req in enumerate(requests):
            version_groups.setdefault(req.model_version, []).append(idx)

        for model_version, indices in version_groups.items():
            t_start = time.perf_counter()
            
            # Setup group targets
            group_requests = [requests[i] for i in indices]

            # 1. Resolve and load the model (checks lifecycle eligibility once per version block)
            try:
                model, cache_hit = self.cache.get_model(model_version)
                model_record = self.registry.repo.get(model_version)
                if not model_record:
                    raise ModelNotFoundError(f"Model version '{model_version}' not found.")
            except Exception as error:
                # Group-level resolution failure. Apply error to all indices in this group.
                t_end = time.perf_counter()
                latency = (t_end - t_start) * 1000.0
                
                if not isinstance(error, InferenceError):
                    error = RegistryError(f"Failed cache model loading: {str(error)}")
                
                for idx_pos, i in enumerate(indices):
                    results[i] = error
                    self.telemetry_sink.record_failure(
                        request=requests[i],
                        error_type=type(error).__name__,
                        error_message=str(error),
                        latency_ms=latency,
                    )
                continue

            # 2. Iterate and validate request features individually to support partial failure isolation
            valid_indices: List[int] = []
            valid_aligned_features: List[List[float]] = []

            for i in indices:
                req = requests[i]
                req_start = time.perf_counter()
                try:
                    aligned = validate_features(
                        features=req.features,
                        model_feature_columns=model.feature_columns,
                        request_feature_version=req.feature_version,
                        expected_feature_version=model_record.feature_version,
                    )
                    valid_indices.append(i)
                    valid_aligned_features.append(aligned)
                except SchemaValidationError as sve:
                    # Single request validation failure: isolate and telemetry this mismatch
                    req_end = time.perf_counter()
                    results[i] = sve
                    self.telemetry_sink.record_failure(
                        request=req,
                        error_type=type(sve).__name__,
                        error_message=str(sve),
                        latency_ms=(req_end - req_start) * 1000.0,
                    )

            if not valid_indices:
                continue

            # 3. Perform batch predictions using PredictionModel optimized 2D shape executor
            # We construct a 2D array: (n_valid, n_features)
            X_batch = np.array(valid_aligned_features)
            try:
                pred_slice = model.predict_batch(X_batch)
                predictions = pred_slice.tolist()
            except Exception as e:
                # Prediction failed for all valid requests in group
                t_end = time.perf_counter()
                latency = (t_end - t_start) * 1000.0
                pred_err = PredictionError(f"Batch prediction execution failed: {str(e)}")
                
                for i in valid_indices:
                    results[i] = pred_err
                    self.telemetry_sink.record_failure(
                        request=requests[i],
                        error_type=type(pred_err).__name__,
                        error_message=str(pred_err),
                        latency_ms=latency,
                    )
                continue

            # 4. Handle classification probabilities batch
            proba_slice = None
            if model.task_type == "classification":
                if hasattr(model, "predict_proba_batch"):
                    try:
                        proba_slice = model.predict_proba_batch(X_batch)
                    except Exception:
                        pass
                
                if proba_slice is None and hasattr(model.raw_model, "predict_proba"):
                    try:
                        proba_slice = model.raw_model.predict_proba(X_batch)
                    except Exception:
                        pass

            # 5. Populate successful results and log telemetry/drift hooks
            for idx_in_valid, i in enumerate(valid_indices):
                req = requests[i]
                pred_val = float(predictions[idx_in_valid])
                
                probabilities = None
                confidence = None
                confidence_source = "UNAVAILABLE"
                is_calibrated = False

                if model.task_type == "classification" and proba_slice is not None:
                    calibrator = getattr(model, "calibrator", None)
                    if calibrator is not None:
                        try:
                            # Apply calibration to the specific slice row
                            calibrated_slice = calibrator.transform(proba_slice)
                            row = calibrated_slice[idx_in_valid]
                            probabilities = row.tolist()
                            confidence = float(np.max(row))
                            calib_meta = getattr(model_record, "calibration_metadata", {})
                            calib_method = calib_meta.get("calibration_method", "calibrated")
                            confidence_source = calib_method.upper()
                            is_calibrated = True
                        except Exception as e:
                            # Fail-closed
                            raise PredictionError(f"Batch probability calibration failed: {str(e)}") from e
                    else:
                        row = proba_slice[idx_in_valid]
                        probabilities = row.tolist()
                        confidence = float(np.max(row))
                        confidence_source = "UNCALIBRATED"

                # Drift hooks
                for observer in self.drift_observers:
                    try:
                        observer.on_observation(model_version, req.features)
                    except Exception:
                        pass

                t_end = time.perf_counter()
                latency = (t_end - t_start) * 1000.0  # Overall chunk timing

                response = InferenceResponse(
                    model_version=model_version,
                    symbol=req.symbol,
                    prediction=pred_val,
                    probabilities=probabilities,
                    confidence=confidence,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    latency_ms=latency,
                    feature_version=model_record.feature_version,
                    metadata=InferenceMetadata(
                        cache_hit=cache_hit,
                        is_calibrated=is_calibrated,
                        confidence_source=confidence_source,
                    ),
                )
                results[i] = response
                self.telemetry_sink.record_success(req, response)

        return results



