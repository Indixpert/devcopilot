import logging
from typing import Dict, Any, Optional

from .base import BaseLLMProvider
from ..core.config import settings

logger = logging.getLogger(__name__)


def def get_provider(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> BaseLLMProvider:
    """
    Factory function to get an LLM provider instance.

    The provider is selected based on the 'name' argument, or the
    LLM_PROVIDER setting if 'name' is not provided.

    Configuration is passed via the 'config' dict. If not provided,
    it's inferred from environment variables within the provider's __init__.
    """
    provider_name = (name or settings.LLM_PROVIDER).lower()
    config = config or {}

    logger.info(f"Attempting to initialize LLM provider: {provider_name}")

    if provider_name == "openai":
        from .openai_provider import OpenAIProvider
        provider_class = OpenAIProvider
    elif provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        provider_class = AnthropicProvider
    elif provider_name == "ollama":
        from .ollama_provider import OllamaProvider
        provider_class = OllamaProvider
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

    # The config dict allows overriding env vars for specific instantiations.
    # The providers themselves are responsible for reading env vars if config is not passed.
    return provider_class(**config)
