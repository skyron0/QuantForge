"""
Feature Runtime service orchestrator.

Coordinates:
  buffer → extractor → validator → inference → signal mapper

Returns an immutable FeatureRuntimeResult with stage timings and status.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from backend.feature_runtime.policy import FeatureRuntimePolicy
from backend.feature_runtime.schema import FeatureSchema
from backend.feature_runtime.buffer import HistoricalFeatureBuffer
from backend.feature_runtime.extractor import FeatureExtractor
from backend.feature_runtime.validator import FeatureValidator
from backend.feature_runtime.inference import FeatureInferenceEngine
from backend.feature_runtime.signal import FeatureSignalMapper
from backend.feature_runtime.telemetry import FeatureRuntimeTelemetrySink
from backend.feature_runtime.models import (
    FeatureRuntimeResult,
    FeatureRuntimeStatus,
    FeatureSnapshot,
    PredictionResult,
)
from backend.feature_runtime.exceptions import (
    InsufficientHistoryError,
    FeatureWarmupError,
    StaleFeatureError,
    FeatureSchemaMismatchError,
    FeatureOrderingError,
    MissingFeatureError,
    InvalidFeatureValueError,
    ModelCompatibilityError,
    InferenceRuntimeError,
    ModelUnavailableError,
    InvalidPredictionError,
    SignalGenerationError,
)
from backend.decision.models import MLSignal


class FeatureRuntimeService:
    """
    Top-level coordinator for the Feature Runtime pipeline.

    Produces only information-layer outputs: FeatureSnapshot,
    PredictionResult, MLSignal.  Never touches execution, risk,
    or order intent.
    """

    def __init__(
        self,
        policy: FeatureRuntimePolicy,
        schema: FeatureSchema,
        buffer: HistoricalFeatureBuffer,
        extractor: FeatureExtractor,
        validator: FeatureValidator,
        inference_engine: FeatureInferenceEngine,
        signal_mapper: FeatureSignalMapper,
        telemetry_sink: Optional[FeatureRuntimeTelemetrySink] = None,
    ) -> None:
        self.policy = policy
        self.schema = schema
        self.buffer = buffer
        self.extractor = extractor
        self.validator = validator
        self.inference_engine = inference_engine
        self.signal_mapper = signal_mapper
        self.telemetry = telemetry_sink

    # ── public ────────────────────────────────────────────────────────────

    def process(self, symbol: str, timestamp: str) -> FeatureRuntimeResult:
        """
        Run the full pipeline for *symbol* at *timestamp*.
        """
        start = time.perf_counter()
        stage_timings: dict = {}

        snapshot: Optional[FeatureSnapshot] = None
        prediction: Optional[PredictionResult] = None
        signal: Optional[MLSignal] = None

        # ── 1. Staleness check ────────────────────────────────────────────
        stale_start = time.perf_counter()
        try:
            self._check_staleness(timestamp)
            stage_timings["staleness_check"] = (time.perf_counter() - stale_start) * 1000.0
        except StaleFeatureError as exc:
            stage_timings["staleness_check"] = (time.perf_counter() - stale_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.STALE_DATA,
                "STALENESS", str(exc), stage_timings, start,
            )

        # ── 2. Feature extraction ─────────────────────────────────────────
        extract_start = time.perf_counter()
        try:
            snapshot = self.extractor.extract(symbol, timestamp)
            stage_timings["extraction"] = (time.perf_counter() - extract_start) * 1000.0
        except (InsufficientHistoryError, FeatureWarmupError) as exc:
            stage_timings["extraction"] = (time.perf_counter() - extract_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.WARMUP_SKIP,
                "EXTRACTION", str(exc), stage_timings, start,
            )
        except (InvalidFeatureValueError, MissingFeatureError) as exc:
            stage_timings["extraction"] = (time.perf_counter() - extract_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.VALIDATION_FAILED,
                "EXTRACTION", str(exc), stage_timings, start,
            )

        # ── 3. Validation ─────────────────────────────────────────────────
        val_start = time.perf_counter()
        try:
            self.validator.validate(snapshot)
            stage_timings["validation"] = (time.perf_counter() - val_start) * 1000.0
        except (FeatureSchemaMismatchError,) as exc:
            stage_timings["validation"] = (time.perf_counter() - val_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.SCHEMA_MISMATCH,
                "VALIDATION", str(exc), stage_timings, start,
            )
        except (FeatureOrderingError, MissingFeatureError, InvalidFeatureValueError) as exc:
            stage_timings["validation"] = (time.perf_counter() - val_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.VALIDATION_FAILED,
                "VALIDATION", str(exc), stage_timings, start,
            )

        # ── 4. Inference ──────────────────────────────────────────────────
        inf_start = time.perf_counter()
        try:
            prediction = self.inference_engine.predict(snapshot)
            stage_timings["inference"] = (time.perf_counter() - inf_start) * 1000.0
        except (ModelCompatibilityError,) as exc:
            stage_timings["inference"] = (time.perf_counter() - inf_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.SCHEMA_MISMATCH,
                "INFERENCE", str(exc), stage_timings, start,
            )
        except (InferenceRuntimeError, ModelUnavailableError, InvalidPredictionError) as exc:
            stage_timings["inference"] = (time.perf_counter() - inf_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.INFERENCE_FAILED,
                "INFERENCE", str(exc), stage_timings, start,
            )

        # ── 5. Signal mapping ─────────────────────────────────────────────
        sig_start = time.perf_counter()
        try:
            signal = self.signal_mapper.map(prediction)
            stage_timings["signal"] = (time.perf_counter() - sig_start) * 1000.0
        except SignalGenerationError as exc:
            stage_timings["signal"] = (time.perf_counter() - sig_start) * 1000.0
            return self._fail(
                symbol, timestamp, FeatureRuntimeStatus.SIGNAL_FAILED,
                "SIGNAL", str(exc), stage_timings, start,
            )

        # ── success ───────────────────────────────────────────────────────
        total_ms = (time.perf_counter() - start) * 1000.0
        result = FeatureRuntimeResult(
            status=FeatureRuntimeStatus.SUCCESS,
            symbol=symbol,
            timestamp=timestamp,
            feature_snapshot=snapshot,
            prediction_result=prediction,
            ml_signal=signal,
            stage_timings=stage_timings,
            total_latency_ms=total_ms,
        )
        if self.telemetry:
            self.telemetry.record_result(result)
        return result

    # ── internal ──────────────────────────────────────────────────────────

    def _check_staleness(self, timestamp: str) -> None:
        """Fail closed if data timestamp is too old."""
        try:
            ts = timestamp
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
        except Exception:
            return  # Non-parseable timestamps pass (caution)

        now = datetime.now(timezone.utc)
        age = (now - dt).total_seconds()
        if age > self.policy.staleness_limit_seconds:
            raise StaleFeatureError(
                f"Data is {age:.1f}s old (limit {self.policy.staleness_limit_seconds}s)"
            )

    def _fail(
        self,
        symbol: str,
        timestamp: str,
        status: FeatureRuntimeStatus,
        stage: str,
        reason: str,
        stage_timings: dict,
        start: float,
    ) -> FeatureRuntimeResult:
        total_ms = (time.perf_counter() - start) * 1000.0
        result = FeatureRuntimeResult(
            status=status,
            symbol=symbol,
            timestamp=timestamp,
            failed_stage=stage,
            rejection_reason=reason,
            stage_timings=stage_timings,
            total_latency_ms=total_ms,
        )
        if self.telemetry:
            self.telemetry.record_result(result)
        return result
