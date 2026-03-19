import os
import time
import json
import logging
from typing import AsyncIterator, List, Dict, Any

import httpx

from .base import BaseLLMProvider, ProviderError, rate_limit_retry_decorator

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        self._model_checked = False

    async def _ensure_model_is_available(self):
        if self._model_checked:
            return
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            if not any(m["name"].startswith(self.model) for m in models):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Attempting to pull it. This may take a while..."
                )
                await self._pull_model()
            self._model_checked = True
        except httpx.RequestError as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}. Is it running?")
            raise ProviderError(f"Could not connect to Ollama: {e}") from e

    async def _pull_model(self):
        try:
            async with self.client.stream(
                "POST", "/api/pull", json={"name": self.model, "stream": True}, timeout=300.0
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            logger.info(f"Pulling {self.model}: {data['status']}")
                        if "error" in data:
                            raise ProviderError(f"Failed to pull model: {data['error']}")
            logger.info(f"Successfully pulled model '{self.model}'")
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"Failed to pull model '{self.model}': {e.response.text}") from e

    @rate_limit_retry_decorator
    async def complete(self, prompt: str, **kwargs: Dict[str, Any]) -> AsyncIterator[str]:
        await self._ensure_model_is_available()
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }

        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done"):
                            input_tokens = chunk.get("prompt_eval_count", 0)
                            output_tokens = chunk.get("eval_count", 0)
                            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise ProviderError("Ollama rate limit exceeded") from e
            logger.error("Ollama completion error: %s - %s", e, e.response.text)
            raise
        except Exception as e:
            logger.error("Ollama completion error: %s", e, exc_info=True)
            raise

        latency = time.time() - start_time
        logger.info(
            "Ollama completion call",
            extra={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": int(latency * 1000),
            },
        )

    @rate_limit_retry_decorator
    async def embed(self, text: str) -> List[float]:
        await self._ensure_model_is_available()
        start_time = time.time()

        payload = {"model": self.model, "prompt": text}

        try:
            response = await self.client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise ProviderError("Ollama rate limit exceeded") from e
            logger.error("Ollama embedding error: %s - %s", e, e.response.text)
            raise
        except Exception as e:
            logger.error("Ollama embedding error: %s", e, exc_info=True)
            raise
        finally:
            latency = time.time() - start_time
            logger.info(
                "Ollama embedding call",
                extra={
                    "model": self.model,
                    "input_tokens": 0,  # Ollama embedding API doesn't return token counts
                    "latency_ms": int(latency * 1000),
                },
            )
