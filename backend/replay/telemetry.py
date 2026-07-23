import logging
from typing import Dict, Any

class ReplayTelemetrySink:
    """
    Interface for tracking and reporting historical simulation statistics.
    """
    def record_step(self, step_index: int, total_steps: int, details: Dict[str, Any]) -> None:
        pass

    def record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        pass

class ConsoleReplayTelemetrySink(ReplayTelemetrySink):
    """
    Console logging implementation of replay telemetry sink.
    """
    def __init__(self, logger_name: str = "QuantForge.ReplayTelemetry") -> None:
        self.logger = logging.getLogger(logger_name)

    def record_step(self, step_index: int, total_steps: int, details: Dict[str, Any]) -> None:
        pct = (step_index / total_steps) * 100.0 if total_steps > 0 else 0.0
        self.logger.info(
            f"[REPLAY PROGRESS] Step {step_index}/{total_steps} ({pct:.1f}%) | "
            f"Symbol: {details.get('symbol')} | Close: {details.get('close')}"
        )

    def record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        self.logger.info(f"[REPLAY EVENT] {event_type} | Details: {details}")
