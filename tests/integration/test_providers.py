import os
import pytest
from unittest.mock import patch, AsyncMock

import openai

from backend.app.llm.factory import get_provider
from backend.app.llm.base import BaseLLMProvider, ProviderError

# --- Fixtures for each provider ---

@pytest.fixture(scope="module")
def openai_provider():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping OpenAI tests.")
    return get_provider("openai")

@pytest.fixture(scope="module")
def anthropic_provider():
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY or VOYAGE_API_KEY not set, skipping Anthropic tests.")
    return get_provider("anthropic")

@pytest.fixture(scope="module")
def ollama_provider():
    if os.getenv("SKIP_OLLAMA_TESTS"):
        pytest.skip("Skipping Ollama tests.")
    try:
        provider = get_provider("ollama", config={"model": "phi3"}) # use a small model for tests
        import httpx
        try:
            # Use the client from the provider to respect configured base_url
            httpx.get(str(provider.client.base_url))
        except httpx.ConnectError:
            pytest.skip("Ollama not running, skipping tests.")
        return provider
    except Exception as e:
        pytest.skip(f"Failed to initialize Ollama provider: {e}")

# --- Parametrized tests ---

@pytest.mark.parametrize(
    "provider_fixture",
    ["openai_provider", "anthropic_provider", "ollama_provider"],
)
@pytest.mark.asyncio
async def test_provider_complete_streams_tokens(provider_fixture, request):
    """Test that the complete method streams at least one token."""
    provider: BaseLLMProvider = request.getfixturevalue(provider_fixture)
    prompt = "Hello, world! Tell me a short story about a robot."
    
    content = ""
    token_count = 0
    async for token in provider.complete(prompt):
        assert isinstance(token, str)
        content += token
        token_count += 1
    
    assert token_count > 0
    assert len(content) > 0
    print(f"Provider {type(provider).__name__} generated content: {content[:80]}...")

@pytest.mark.parametrize(
    "provider_fixture",
    ["openai_provider", "anthropic_provider", "ollama_provider"],
)
@pytest.mark.asyncio
async def test_provider_embed_returns_float_list(provider_fixture, request):
    """Test that the embed method returns a list of floats."""
    provider: BaseLLMProvider = request.getfixturevalue(provider_fixture)
    text = "This is a test sentence for embedding."
    
    embedding = await provider.embed(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    print(f"Provider {type(provider).__name__} produced embedding of dimension {len(embedding)}")

# --- Mocked retry test ---

@pytest.mark.asyncio
async def test_retry_on_rate_limit():
    """Test that the retry decorator correctly handles rate limit errors."""
    from backend.app.llm.openai_provider import OpenAIProvider
    
    mock_client = AsyncMock()
    
    # The successful call should return an async iterator
    successful_stream = AsyncMock()
    successful_stream.__aiter__.return_value = [
        AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="Success"))])
    ]
    
    # Simulate RateLimitError on first 2 calls, then succeed
    mock_client.chat.completions.create.side_effect = [
        openai.RateLimitError("Rate limit", response=AsyncMock(), body=None),
        openai.RateLimitError("Rate limit", response=AsyncMock(), body=None),
        successful_stream
    ]

    with patch("backend.app.llm.openai_provider.AsyncOpenAI", return_value=mock_client):
        # We need to patch tenacity's sleep to speed up the test
        with patch("tenacity.asyncio.AsyncRetrying.sleep", new_callable=AsyncMock) as mock_sleep:
            provider = OpenAIProvider(api_key="dummy")
            
            # This should trigger retries and eventually succeed
            tokens = [token async for token in provider.complete("test prompt")]
            
            assert tokens == ["Success"]
            # Check that it was called 3 times (1 initial + 2 retries)
            assert mock_client.chat.completions.create.call_count == 3
            # Check that we slept twice between retries
            assert mock_sleep.call_count == 2
