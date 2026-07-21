import os
import json
import pytest
from unittest.mock import MagicMock, patch
import requests

from backend.intelligence.exceptions import (
    ProviderConfigurationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderResponseError,
    StructuredOutputError,
    ReasoningError,
)
from backend.intelligence.models import (
    AIRequest,
    AIResponse,
    ReasoningRequest,
    ReasoningResult,
    ProviderHealth,
)
from backend.intelligence.providers.base import BaseAIProvider
from backend.intelligence.providers.registry import AIProviderRegistry
from backend.intelligence.providers.ollama_provider import OllamaProvider
from backend.intelligence.reasoning.prompts import get_prompt_template
from backend.intelligence.reasoning.engine import ReasoningEngine
from backend.intelligence.telemetry import AITelemetry


# =====================================================================
# Mock Provider for Testing
# =====================================================================
class DummyProvider(BaseAIProvider):
    """Simple provider mock to test registry and abstract contracts."""

    def __init__(self, config):
        self._model_name = config.get("model_name")
        self.responses = []
        self.calls = []

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            available=True, provider="dummy", model=self._model_name, latency_ms=1.0
        )

    def generate(self, request: AIRequest) -> AIResponse:
        self.calls.append(request)
        resp = self.responses.pop(0)
        resp.request_id = request.request_id
        return resp

    def generate_structured(self, request: AIRequest) -> AIResponse:
        self.calls.append(request)
        resp = self.responses.pop(0)
        resp.request_id = request.request_id
        return resp

    @property
    def provider_name(self) -> str:
        return "dummy"

    @property
    def model_name(self) -> str:
        return self._model_name


# =====================================================================
# Unit Tests
# =====================================================================

def test_registry_resolution():
    """Verify provider registration, creation, and config validation."""
    AIProviderRegistry.register("dummy", DummyProvider)

    # 1. Successful creation
    config = {"model_name": "llama3"}
    provider = AIProviderRegistry.create("dummy", config)
    assert isinstance(provider, DummyProvider)
    assert provider.model_name == "llama3"

    # 2. Check capitalization case-insensitive resolution
    provider_upper = AIProviderRegistry.create("DUMMY", config)
    assert isinstance(provider_upper, DummyProvider)

    # 3. Missing model_name must raise ProviderConfigurationError
    with pytest.raises(ProviderConfigurationError) as exc_info:
        AIProviderRegistry.create("dummy", {})
    assert "model_name is not configured" in str(exc_info.value)

    # 4. Unknown provider must raise ProviderConfigurationError
    with pytest.raises(ProviderConfigurationError) as exc_info_unk:
        AIProviderRegistry.create("nonexistent", config)
    assert "Unknown or unsupported AI provider" in str(exc_info_unk.value)


@patch("requests.get")
def test_ollama_provider_health_check(mock_get):
    """Verify OllamaProvider health check behaviors (success, failure, model presence)."""
    # Configure OllamaProvider
    config = {"model_name": "llama3", "base_url": "http://localhost:11434"}
    prov = OllamaProvider(config)

    # CASE A: Success running, model exists
    mock_root = MagicMock()
    mock_root.status_code = 200
    mock_tags = MagicMock()
    mock_tags.status_code = 200
    mock_tags.json.return_value = {"models": [{"name": "llama3:latest"}, {"name": "mistral"}]}
    mock_get.side_effect = [mock_root, mock_tags]

    health = prov.health_check()
    assert health.available is True
    assert health.provider == "ollama"
    assert health.model == "llama3"
    assert health.error is None

    # CASE B: Success running, model does NOT exist
    mock_get.side_effect = None
    mock_root.status_code = 200
    mock_tags.status_code = 200
    mock_tags.json.return_value = {"models": [{"name": "mistral"}]}
    mock_get.side_effect = [mock_root, mock_tags]

    health = prov.health_check()
    assert health.available is False
    assert health.error is not None and "not downloaded" in health.error

    # CASE C: Service unreachable / status non-200
    mock_get.side_effect = None
    mock_root.status_code = 500
    mock_get.side_effect = [mock_root]
    health = prov.health_check()
    assert health.available is False
    assert health.error is not None and "Unhealthy root" in health.error

    # CASE D: Connection timeout
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
    health = prov.health_check()
    assert health.available is False
    assert health.error is not None and "timeout" in health.error.lower()


@patch("requests.post")
def test_ollama_provider_generation(mock_post):
    """Verify OllamaProvider standard and structured request posting."""
    config = {"model_name": "llama3", "base_url": "http://localhost:11434"}
    prov = OllamaProvider(config)

    # Standard JSON mock response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Hello World!"}
    mock_post.return_value = mock_resp

    req = AIRequest(
        request_id="req-123",
        task_type="info",
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.0
    )

    # 1. Standard text generation
    res = prov.generate(req)
    assert res.success is True
    assert res.content == "Hello World!"
    # Verify post payloads
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:11434/api/generate"
    assert kwargs["json"]["model"] == "llama3"
    assert kwargs["json"]["prompt"] == "usr"
    assert "format" not in kwargs["json"]

    # 2. Structured JSON generation
    mock_resp.json.return_value = {"response": '{"val": 42}'}
    res_struct = prov.generate_structured(req)
    assert res_struct.success is True
    assert res_struct.structured_output == {"val": 42}
    # Verify format parameter in payload
    _, kwargs_struct = mock_post.call_args
    assert kwargs_struct["json"]["format"] == "json"


@patch("requests.post")
def test_ollama_exceptions_translation(mock_post):
    """Verify translation of connection errors, timeouts, and server errors."""
    config = {"model_name": "llama3", "base_url": "http://localhost:11434"}
    prov = OllamaProvider(config)
    req = AIRequest(request_id="1", task_type="t", system_prompt="s", user_prompt="u")

    # Timeout
    mock_post.side_effect = requests.exceptions.Timeout("Timeout!")
    with pytest.raises(ProviderTimeoutError):
        prov.generate(req)

    # Connection Error
    mock_post.side_effect = requests.exceptions.ConnectionError("Offline")
    with pytest.raises(ProviderUnavailableError):
        prov.generate(req)

    # Status Code 500
    mock_post.side_effect = None
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal error"
    mock_post.return_value = mock_resp
    with pytest.raises(ProviderResponseError):
        prov.generate(req)


def test_structured_validation_pipeline():
    """Verify details of the response verification pipeline in ReasoningEngine."""
    prov = DummyProvider({"model_name": "mock-model"})
    engine = ReasoningEngine(provider=prov, max_retries=3)

    req = ReasoningRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        timestamp="2026-07-21T12:00:00Z",
        features={"rsi": 35.0},
        ml_predictions={"trend": 1.0},
        market_context={},
        portfolio_context={},
        risk_context={}
    )

    # CASE 1: Perfect Output
    valid_output = {
        "market_regime": "trending",
        "directional_bias": "bullish",
        "confidence": 0.85,
        "risk_flags": ["low_volume"],
        "reasoning_summary": "ML predicts trend and RSI is near oversold levels.",
        "evidence": ["rsi=35.0", "ml_pred=1"]
    }
    prov.responses = [
        AIResponse(
            request_id="req-ok",
            provider="dummy",
            model="mock-model",
            content=json.dumps(valid_output),
            structured_output=valid_output,
            success=True
        )
    ]

    res = engine.reason(req)
    assert isinstance(res, ReasoningResult)
    assert res.market_regime == "trending"
    assert res.directional_bias == "bullish"
    assert res.confidence == 0.85
    assert res.provider == "dummy"
    assert res.model == "mock-model"
    assert res.prompt_id == "market_reasoning"
    assert res.prompt_version == "v1"

    # CASE 2: Invalid JSON Parsing
    prov.responses = [
        AIResponse(
            request_id="req-bad-json",
            provider="dummy",
            model="mock-model",
            content="invalid JSON content {{{",
            structured_output=None,
            success=True
        )
    ] * 3  # Fill for all 3 retries
    with pytest.raises(ReasoningError) as exc_info:
        engine.reason(req)
    assert "failed after 3 retries" in str(exc_info.value)
    # The cause should be StructuredOutputError
    assert isinstance(exc_info.value.__cause__, StructuredOutputError)

    # CASE 3: Schema Violation (e.g. missing reasoning_summary)
    invalid_schema = valid_output.copy()
    del invalid_schema["reasoning_summary"]
    prov.responses = [
        AIResponse(
            request_id="req-bad-schema",
            provider="dummy",
            model="mock-model",
            content=json.dumps(invalid_schema),
            structured_output=invalid_schema,
            success=True
        )
    ] * 3
    with pytest.raises(ReasoningError) as exc_info_schema:
        engine.reason(req)
    assert "Schema validation failed" in str(exc_info_schema.value.__cause__)

    # CASE 4: Domain Confidence Violation (confidence out of bounds)
    bad_confidence = valid_output.copy()
    bad_confidence["confidence"] = 1.05
    prov.responses = [
        AIResponse(
            request_id="req-bad-conf",
            provider="dummy",
            model="mock-model",
            content=json.dumps(bad_confidence),
            structured_output=bad_confidence,
            success=True
        )
    ] * 3
    with pytest.raises(ReasoningError) as exc_info_conf:
        engine.reason(req)
    assert "confidence must be between 0.0 and 1.0" in str(exc_info_conf.value.__cause__)

    # CASE 5: Retry Recovery (Attempt 1 fails, Attempt 2 succeeds)
    prov.responses = [
        AIResponse(
            request_id="req-retry-1",
            provider="dummy",
            model="mock-model",
            content="broken json",
            success=True
        ),
        AIResponse(
            request_id="req-retry-2",
            provider="dummy",
            model="mock-model",
            content=json.dumps(valid_output),
            structured_output=valid_output,
            success=True
        )
    ]
    res_recovered = engine.reason(req)
    assert res_recovered.confidence == 0.85
    assert prov.calls[-1].request_id == res_recovered.request_id


def test_architecture_import_boundary():
    """Verify that backend/intelligence does not import execution/broker/exchange modules."""
    forbidden_imports = [
        "backend.execution",
        "backend.broker",
        "PaperExecutor",
        "LiveExecutor",
        "OrderExecutor",
        "MarketConsumer",
    ]
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "backend", "intelligence")
    )
    assert os.path.exists(base_dir), f"Intelligence package path not found at {base_dir}"

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    for forbidden in forbidden_imports:
                        # Confirm imports do not appear in code text
                        assert forbidden not in content, (
                            f"Architecture safety violation: Forbidden token '{forbidden}' "
                            f"found in reasoning module: {filepath}"
                        )
