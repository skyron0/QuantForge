import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from backend.inference.models import InferenceRequest, InferenceResponse

logger = logging.getLogger("backend.inference.telemetry")


class InferenceTelemetrySink(ABC):
    """
    Abstract contract for recording inference events (successes and failures).
    Decouples InferenceEngine from specific monitoring tools like Prometheus or OpenTelemetry.
    """

    @abstractmethod
    def record_success(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
    ) -> None:
        """
        Record a successful model prediction event.
        """
        pass

    @abstractmethod
    def record_failure(
        self,
        request: InferenceRequest,
        error_type: str,
        error_message: str,
        latency_ms: float,
    ) -> None:
        """
        Record an inference failure event.
        """
        pass


class LoggerTelemetrySink(InferenceTelemetrySink):
    """
    Default structured logging implementation of InferenceTelemetrySink.
    Logs structured outputs as JSON lines for easy aggregation.
    """

    def record_success(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
    ) -> None:
        log_payload = {
            "event": "inference_success",
            "model_version": response.model_version,
            "symbol": response.symbol,
            "prediction": response.prediction,
            "confidence": response.confidence,
            "latency_ms": response.latency_ms,
            "cache_hit": response.metadata.cache_hit,
            "confidence_source": response.metadata.confidence_source,
            "timestamp": response.timestamp,
            "feature_version": response.feature_version,
        }
        logger.info(json.dumps(log_payload))

    def record_failure(
        self,
        request: InferenceRequest,
        error_type: str,
        error_message: str,
        latency_ms: float,
    ) -> None:
        log_payload = {
            "event": "inference_failure",
            "model_version": request.model_version,
            "symbol": request.symbol,
            "error_type": error_type,
            "error_message": error_message,
            "latency_ms": latency_ms,
            "timestamp": request.timestamp,
        }
        logger.error(json.dumps(log_payload))
