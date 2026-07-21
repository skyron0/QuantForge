import logging

logger = logging.getLogger("QuantForge.Intelligence.Telemetry")


class AITelemetry:
    """
    Telemetry recording system for intelligence reasoning pipelines.
    Tracks performance indicators and health signals without exporting keys.
    """

    def record_success(
        self,
        request_id: str,
        provider: str,
        model: str,
        task_type: str,
        latency_ms: float,
        prompt_version: str,
        prompt_id: str
    ) -> None:
        """Log a successful AI request invocation event in a secure audit format."""
        logger.info(
            f"[AI Telemetry Success] RequestID: {request_id} | Provider: {provider} | "
            f"Model: {model} | Task: {task_type} | Latency: {latency_ms:.2f}ms | "
            f"Prompt: {prompt_id} (v{prompt_version})"
        )

    def record_failure(
        self,
        request_id: str,
        provider: str,
        model: str,
        task_type: str,
        error_type: str,
        prompt_version: str,
        prompt_id: str
    ) -> None:
        """Log a failed AI request invocation event to help troubleshoot."""
        logger.error(
            f"[AI Telemetry Failure] RequestID: {request_id} | Provider: {provider} | "
            f"Model: {model} | Task: {task_type} | ErrorType: {error_type} | "
            f"Prompt: {prompt_id} (v{prompt_version})"
        )
