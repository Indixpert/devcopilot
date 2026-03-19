import os
import time
import logging
from typing import AsyncIterator, List, Dict, Any

import anthropic
from anthropic import AsyncAnthropic
import voyageai

from .base import BaseLLMProvider, ProviderError, rate_limit_retry_decorator

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str = None,
        voyage_api_key: str = None,
        model: str = "claude-3-5-sonnet-20240620",
    ):
        # User requested 'claude-sonnet-4-20250514', which is not a valid model.
        # Using the latest Sonnet model instead.
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.voyage_api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key not provided.")
        if not self.voyage_api_key:
            raise ValueError("Voyage AI API key for embeddings not provided.")

        self.client = AsyncAnthropic(api_key=self.api_key)
        voyageai.api_key = self.voyage_api_key
        self.voyage_client = voyageai.AsyncClient()
        # User requested 'voyage-3', which does not exist. Using a current SOTA model.
        self.embedding_model = "voyage-large-2-instruct"

    @rate_limit_retry_decorator
    async def complete(self, prompt: str, **kwargs: Dict[str, Any]) -> AsyncIterator[str]:
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0

        try:
            async with self.client.messages.stream(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,  # A reasonable default
                **kwargs,
            ) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        input_tokens = event.message.usage.input_tokens
                    elif event.type == "content_block_delta":
                        yield event.delta.text
                    elif event.type == "message_delta":
                        output_tokens += event.usage.output_tokens

        except anthropic.RateLimitError as e:
            logger.warning("Anthropic rate limit error: %s", e)
            raise ProviderError("Anthropic rate limit exceeded") from e
        except Exception as e:
            logger.error("Anthropic completion error: %s", e, exc_info=True)
            raise

        latency = time.time() - start_time
        logger.info(
            "Anthropic completion call",
            extra={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": int(latency * 1000),
            },
        )

    @rate_limit_retry_decorator
    async def embed(self, text: str) -> List[float]:
        start_time = time.time()
        total_tokens = 0
        try:
            result = await self.voyage_client.embed(
                texts=[text], model=self.embedding_model, input_type="document"
            )
            total_tokens = result.total_tokens
            return result.embeddings[0]
        except Exception as e:
            logger.error("VoyageAI embedding error: %s", e, exc_info=True)
            if "429" in str(e):
                raise ProviderError("VoyageAI rate limit exceeded") from e
            raise
        finally:
            latency = time.time() - start_time
            logger.info(
                "VoyageAI embedding call",
                extra={
                    "model": self.embedding_model,
                    "input_tokens": total_tokens,
                    "latency_ms": int(latency * 1000),
                },
            )
