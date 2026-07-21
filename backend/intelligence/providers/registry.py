from typing import Dict, Any, Type
from backend.intelligence.providers.base import BaseAIProvider
from backend.intelligence.exceptions import ProviderConfigurationError


class AIProviderRegistry:
    """Registry to map and instantiate concrete BaseAIProvider implementations."""

    _registry: Dict[str, Type[BaseAIProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[BaseAIProvider]) -> None:
        """Register a new AI provider class with a unique handle name."""
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def create(cls, provider_name: str, config: Dict[str, Any]) -> BaseAIProvider:
        """
        Create and return an instance of a registered AI provider.
        Raises ProviderConfigurationError if provider_name is unregistered, or
        if configuration is invalid (e.g. absent model configuration).
        """
        provider_key = provider_name.lower().strip()
        if provider_key not in cls._registry:
            raise ProviderConfigurationError(
                f"Unknown or unsupported AI provider: '{provider_name}'."
            )

        # Validate that the model name is explicitly configured
        model_name = config.get("model_name")
        if not model_name:
            raise ProviderConfigurationError(
                "model_name is not configured. AI_MODEL must be explicitly set."
            )

        provider_cls = cls._registry[provider_key]
        try:
            return provider_cls(config)
        except Exception as e:
            if isinstance(e, ProviderConfigurationError):
                raise e
            raise ProviderConfigurationError(
                f"Failed to initialize provider '{provider_name}': {str(e)}"
            ) from e
