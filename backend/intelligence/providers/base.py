from abc import ABC, abstractmethod
from backend.intelligence.models import AIRequest, AIResponse, ProviderHealth


class BaseAIProvider(ABC):
    """Abstract base class representing a provider-independent AI runtime contract."""

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Check provider status and return a ProviderHealth status object."""
        pass

    @abstractmethod
    def generate(self, request: AIRequest) -> AIResponse:
        """Send a standard text generation request to the provider."""
        pass

    @abstractmethod
    def generate_structured(self, request: AIRequest) -> AIResponse:
        """Send a structured JSON output generation request to the provider."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the registration handle name of the provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the target model name configured on the provider."""
        pass
