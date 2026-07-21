import uuid
import datetime
import logging
from typing import Dict, Any, Optional

from backend.intelligence.models import (
    AIRequest,
    ReasoningRequest,
    ReasoningResult,
)
from backend.intelligence.providers.base import BaseAIProvider
from backend.intelligence.exceptions import (
    StructuredOutputError,
    ReasoningError,
    ProviderError,
)
from backend.intelligence.reasoning.prompts import get_prompt_template
from backend.intelligence.telemetry import AITelemetry

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Orchestration engine evaluating ReasoningRequests using configured BaseAIProvider
    and version-controlled prompt templates. Validates output structure and range bounds.
    """

    def __init__(
        self,
        provider: BaseAIProvider,
        prompt_id: str = "market_reasoning",
        prompt_version: str = "v1",
        max_retries: int = 3,
        telemetry: Optional[AITelemetry] = None
    ):
        self.provider = provider
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.telemetry = telemetry or AITelemetry()

    def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """
        Processes the ReasoningRequest, constructs templates, executes structured query,
        runs validation pipeline and returns validated ReasoningResult.
        """
        # Load the prompt template
        try:
            template = get_prompt_template(self.prompt_id, self.prompt_version)
        except KeyError as e:
            raise ReasoningError(f"Failed to load prompt template: {str(e)}") from e

        # Format system & user prompts
        system_prompt = template["system_template"]
        try:
            user_prompt = template["user_template"].format(
                symbol=request.symbol,
                timeframe=request.timeframe,
                timestamp=request.timestamp,
                features=request.features,
                ml_predictions=request.ml_predictions,
                market_context=request.market_context,
                portfolio_context=request.portfolio_context,
                risk_context=request.risk_context,
                additional_context=request.additional_context or {}
            )
        except Exception as e:
            raise ReasoningError(f"Prompt formatting error: {str(e)}") from e

        response_schema = template["response_schema"]

        # Run with retries for structured parsing/domain errors
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            request_id = str(uuid.uuid4())
            ai_request = AIRequest(
                request_id=request_id,
                task_type=template["task_type"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                response_schema=response_schema
            )

            try:
                # 1. Invoke BaseAIProvider
                response = self.provider.generate_structured(ai_request)

                # 2. Validate Response (parsing, schema, domain)
                parsed = self._validate_response(response, response_schema)

                # Generate high-reproducibility result
                res = ReasoningResult(
                    market_regime=parsed["market_regime"],
                    directional_bias=parsed["directional_bias"],
                    confidence=parsed["confidence"],
                    risk_flags=parsed["risk_flags"],
                    reasoning_summary=parsed["reasoning_summary"],
                    evidence=parsed["evidence"],
                    provider=response.provider,
                    model=response.model,
                    latency_ms=response.latency_ms,
                    request_id=response.request_id,
                    prompt_id=self.prompt_id,
                    prompt_version=self.prompt_version,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
                )

                # Emit success telemetry
                self.telemetry.record_success(
                    request_id=response.request_id,
                    provider=response.provider,
                    model=response.model,
                    task_type=template["task_type"],
                    latency_ms=response.latency_ms,
                    prompt_version=self.prompt_version,
                    prompt_id=self.prompt_id
                )
                return res

            except (StructuredOutputError, ProviderError) as e:
                last_error = e
                logger.warning(
                    f"Reasoning engine attempt {attempt}/{self.max_retries} failed: {str(e)}"
                )
                # Emit failure telemetry for intermediate tries
                self.telemetry.record_failure(
                    request_id=request_id,
                    provider=self.provider.provider_name,
                    model=self.provider.model_name,
                    task_type=template["task_type"],
                    error_type=type(e).__name__,
                    prompt_version=self.prompt_version,
                    prompt_id=self.prompt_id
                )

        # Retries exhausted
        raise ReasoningError(
            f"ReasoningEngine failed after {self.max_retries} retries. Last error: {str(last_error)}"
        ) from last_error

    def _validate_response(self, response: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structured validation pipeline:
        Raw Response -> JSON Parsing -> Response Schema Validation -> Domain Validation -> Dict[str, Any]
        """
        # 1. JSON Parsing
        try:
            if response.structured_output is not None:
                parsed = response.structured_output
            else:
                import json
                parsed = json.loads(response.content)
        except Exception as e:
            raise StructuredOutputError(f"Response is not valid JSON: {str(e)}") from e

        # 2. JSON Schema Validation
        try:
            import jsonschema
            jsonschema.validate(instance=parsed, schema=schema)
        except Exception as e:
            raise StructuredOutputError(f"JSON Schema validation failed: {str(e)}") from e

        # 3. Domain Validation & Types checks
        # market_regime
        market_regime = parsed.get("market_regime")
        if not isinstance(market_regime, str):
            raise StructuredOutputError("market_regime must be a string.")

        # directional_bias
        directional_bias = parsed.get("directional_bias")
        if not isinstance(directional_bias, str):
            raise StructuredOutputError("directional_bias must be a string.")

        # confidence
        confidence = parsed.get("confidence")
        if not isinstance(confidence, (int, float)):
            raise StructuredOutputError("confidence must be a float or integer.")
        confidence_val = float(confidence)
        if not (0.0 <= confidence_val <= 1.0):
            raise StructuredOutputError(
                f"confidence must be between 0.0 and 1.0. Found: {confidence_val}"
            )

        # risk_flags
        risk_flags = parsed.get("risk_flags")
        if not isinstance(risk_flags, list):
            raise StructuredOutputError("risk_flags must be a list.")
        for item in risk_flags:
            if not isinstance(item, str):
                raise StructuredOutputError("all risk_flags items must be strings.")

        # reasoning_summary
        reasoning_summary = parsed.get("reasoning_summary")
        if not isinstance(reasoning_summary, str):
            raise StructuredOutputError("reasoning_summary must be a string.")

        # evidence
        evidence = parsed.get("evidence")
        if not isinstance(evidence, list):
            raise StructuredOutputError("evidence must be a list.")
        for item in evidence:
            if not isinstance(item, str):
                raise StructuredOutputError("all evidence items must be strings.")

        # Re-save normalized confidence type
        parsed["confidence"] = confidence_val
        return parsed
