from backend.intelligence.exceptions import (
    IntelligenceError,
    ProviderError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    ProviderResponseError,
    ProviderConfigurationError,
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

# Automatically register OllamaProvider to the registry on package load
AIProviderRegistry.register("ollama", OllamaProvider)

__all__ = [
    "IntelligenceError",
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ProviderResponseError",
    "ProviderConfigurationError",
    "StructuredOutputError",
    "ReasoningError",
    "AIRequest",
    "AIResponse",
    "ReasoningRequest",
    "ReasoningResult",
    "ProviderHealth",
    "BaseAIProvider",
    "AIProviderRegistry",
    "OllamaProvider",
    "get_prompt_template",
    "ReasoningEngine",
    "AITelemetry",
]
