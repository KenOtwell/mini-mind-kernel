"""Tests for progeny.src.llm_client."""
from __future__ import annotations

import json

import httpx
import pytest

from progeny.src.llm_client import generate, configure, LLMConfig, LLMError, GenerateResult, health_check


@pytest.fixture(autouse=True)
def _use_test_config():
    """Reset to a test config before each test."""
    configure(LLMConfig(host="127.0.0.1", port=19999, timeout_seconds=2.0, retry_attempts=0))
    yield
    configure(LLMConfig())


def _mock_chat_response(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


class TestGenerate:
    @pytest.mark.asyncio
    async def test_successful_response(self):
        """Mocked LLM returns valid content."""
        response_body = _mock_chat_response('{"responses": []}')

        # Use httpx mock via monkeypatch
        import httpx as _httpx

        class MockResponse:
            status_code = 200
            def json(self):
                return response_body
            def raise_for_status(self):
                pass

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                return MockResponse()

        original = _httpx.AsyncClient
        _httpx.AsyncClient = lambda **kw: MockClient()
        try:
            result = await generate([{"role": "user", "content": "test"}])
            assert isinstance(result, GenerateResult)
            assert result.content == '{"responses": []}'
        finally:
            _httpx.AsyncClient = original

    @pytest.mark.asyncio
    async def test_connection_error_raises_llm_error(self):
        """Unreachable server raises LLMError."""
        with pytest.raises(LLMError, match="Connection failed"):
            await generate([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_empty_content_raises_llm_error(self):
        """Empty response content raises LLMError."""
        import httpx as _httpx

        class MockResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": ""}}]}
            def raise_for_status(self):
                pass

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                return MockResponse()

        original = _httpx.AsyncClient
        _httpx.AsyncClient = lambda **kw: MockClient()
        try:
            with pytest.raises(LLMError, match="Empty"):
                await generate([{"role": "user", "content": "test"}])
        finally:
            _httpx.AsyncClient = original


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_unreachable_returns_false(self):
        result = await health_check()
        assert result is False
