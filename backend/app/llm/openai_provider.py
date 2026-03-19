import os
import time
import logging
from typing import AsyncIterator, List, Dict, Any

import openai
from openai import AsyncOpenAI

from .base import BaseLLMProvider, ProviderError, rate_limit_retry_decorator

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"

    @rate_limit_retry_decorator
    async def complete(self, prompt: str, **kwargs: Dict[str, Any]) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        start_time = time.time()
        input_tokens = 0  # Not easily available before call for OpenAI
        output_tokens = 0

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    output_tokens += 1  # A rough estimate
                    yield content
        except openai.RateLimitError as e:
            logger.warning("OpenAI rate limit error: %s", e)
            raise ProviderError("OpenAI rate limit exceeded") from e
        except Exception as e:
            logger.error("OpenAI completion error: %s", e, exc_info=True)
            raise

        latency = time.time() - start_time
        logger.info(
            "OpenAI completion call",
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
        input_tokens = 0
        try:
            text = text.replace("\n", " ")
            response = await self.client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            input_tokens = response.usage.prompt_tokens
            embedding = response.data[0].embedding
            return embedding
        except openai.RateLimitError as e:
            logger.warning("OpenAI rate limit error on embedding: %s", e)
            raise ProviderError("OpenAI rate limit exceeded") from e
        except Exception as e:
            logger.error("OpenAI embedding error: %s", e, exc_info=True)
            raise
        finally:
            latency = time.time() - start_time
            logger.info(
                "OpenAI embedding call",
                extra={
                    "model": self.embedding_model,
                    "input_tokens": input_tokens,
                    "latency_ms": int(latency * 1000),
                },
            )
