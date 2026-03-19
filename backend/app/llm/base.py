import abc
import logging
from typing import AsyncIterator, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

class ProviderError(Exception):
    """Custom exception for provider-related errors, especially for retrying."""
    pass

# Define a retry decorator for rate limit errors.
# Concrete providers should catch their specific rate limit exceptions
# and re-raise them as ProviderError.
rate_limit_retry_decorator = retry(
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ProviderError),
    before_sleep=lambda retry_state: logger.warning(
        "Rate limit or temporary error, retrying in %s seconds... (attempt %s)",
        retry_state.next_action.sleep,
        retry_state.attempt_number,
    ),
)

class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    async def complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Generates a completion for a given prompt, streaming results.
        """
        raise NotImplementedError
        yield

    @abc.abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generates an embedding for a given text.
        """
        raise NotImplementedError
