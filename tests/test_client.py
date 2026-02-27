"""Tests for ContextCacheClient — the Python SDK."""

import json
from unittest.mock import MagicMock, patch

import pytest

from context_cache.client import (
    ContextCacheClient,
    PipelineResult,
    RegisterResult,
    RouteResult,
)


def _mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


class TestClientInit:
    def test_default_base_url(self):
        client = ContextCacheClient()
        assert client.base_url == "http://localhost:8421"

    def test_custom_base_url(self):
        client = ContextCacheClient("http://myserver:9000")
        assert client.base_url == "http://myserver:9000"

    def test_trailing_slash_stripped(self):
        client = ContextCacheClient("http://myserver:9000/")
        assert client.base_url == "http://myserver:9000"

    def test_api_key_set_in_headers(self):
        client = ContextCacheClient(api_key="my-secret")
        assert client._session.headers["X-API-Key"] == "my-secret"

    def test_no_api_key(self):
        client = ContextCacheClient()
        assert "X-API-Key" not in client._session.headers

    def test_context_manager(self):
        with ContextCacheClient() as client:
            assert client is not None


class TestClientHealth:
    @patch("context_cache.client.requests.Session")
    def test_health_call(self, MockSession):
        session = MockSession.return_value
        session.get.return_value = _mock_response({"status": "healthy", "model_loaded": True})
        session.headers = {}

        client = ContextCacheClient()
        result = client.health()
        assert result["status"] == "healthy"
        session.get.assert_called_once()


class TestClientRegisterTools:
    @patch("context_cache.client.requests.Session")
    def test_register_tools(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.post.return_value = _mock_response({
            "tool_id": "merchant",
            "num_tools": 3,
            "cache_hash": "abc123",
            "compile_ms": 1500.0,
            "cache_size_mb": 401.0,
            "status": "compiled",
        })

        client = ContextCacheClient()
        result = client.register_tools("merchant", tools=[{"type": "function"}])

        assert isinstance(result, RegisterResult)
        assert result.tool_id == "merchant"
        assert result.num_tools == 3
        assert result.cache_hash == "abc123"
        assert result.compile_ms == 1500.0


class TestClientRoute:
    @patch("context_cache.client.requests.Session")
    def test_route_returns_route_result(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.post.return_value = _mock_response({
            "selections": [
                {"tool_name": "get_weather", "arguments": {"location": "NYC"}}
            ],
            "confidence": 1.0,
            "raw_response": '<tool_call>\n{"name": "get_weather"}\n</tool_call>',
            "timings": {"total_ms": 150, "cache_hit": True},
        })

        client = ContextCacheClient()
        result = client.route("merchant", "What's the weather?")

        assert isinstance(result, RouteResult)
        assert result.tool_name == "get_weather"
        assert result.arguments == {"location": "NYC"}
        assert result.confidence == 1.0
        assert result.cache_hit is True

    @patch("context_cache.client.requests.Session")
    def test_route_empty_selections(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.post.return_value = _mock_response({
            "selections": [],
            "confidence": 0.0,
            "raw_response": "I don't know",
            "timings": {},
        })

        client = ContextCacheClient()
        result = client.route("merchant", "Random text")
        assert result.tool_name == "unknown"
        assert result.arguments == {}


class TestClientPipeline:
    @patch("context_cache.client.requests.Session")
    def test_pipeline_format_only(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.post.return_value = _mock_response({
            "selected_tool": "gmv_summary",
            "arguments": {"time_period": "last_30_days"},
            "tool_result": {"status": "ok"},
            "llm_context": [{"role": "user", "content": "GMV?"}],
            "llm_response": None,
            "confidence": 1.0,
            "timings": {"total_ms": 200},
        })

        client = ContextCacheClient()
        result = client.pipeline("merchant", "What's my GMV?")

        assert isinstance(result, PipelineResult)
        assert result.tool_name == "gmv_summary"
        assert result.llm_response is None
        assert result.confidence == 1.0

    @patch("context_cache.client.requests.Session")
    def test_pipeline_with_llm(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.post.return_value = _mock_response({
            "selected_tool": "gmv_summary",
            "arguments": {},
            "tool_result": {"gmv": 1000000},
            "llm_context": [],
            "llm_response": "Your GMV is $1,000,000.",
            "confidence": 1.0,
            "timings": {"total_ms": 500, "llm_ms": 300},
        })

        client = ContextCacheClient()
        result = client.pipeline(
            "merchant", "What's my GMV?",
            llm_api_key="sk-test",
            llm_model="claude-sonnet-4-20250514",
        )

        assert result.llm_response == "Your GMV is $1,000,000."

    @patch("context_cache.client.requests.Session")
    def test_pipeline_passes_llm_params(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.post.return_value = _mock_response({
            "selected_tool": "x", "arguments": {}, "tool_result": {},
            "llm_context": [], "llm_response": None, "confidence": 1.0,
            "timings": {},
        })

        client = ContextCacheClient()
        client.pipeline(
            "merchant", "query",
            llm_api_key="sk-test",
            llm_model="gpt-4o",
            llm_system_prompt="Be helpful",
            llm_base_url="https://gateway.com",
            llm_headers={"X-Team": "analytics"},
        )

        # Verify the request body contains all LLM params
        call_args = session.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["llm_api_key"] == "sk-test"
        assert body["llm_model"] == "gpt-4o"
        assert body["llm_system_prompt"] == "Be helpful"
        assert body["llm_base_url"] == "https://gateway.com"
        assert body["llm_headers"] == {"X-Team": "analytics"}


class TestClientListAndRemove:
    @patch("context_cache.client.requests.Session")
    def test_list_tool_sets(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.get.return_value = _mock_response({
            "tool_sets": [
                {"tool_id": "merchant", "num_tools": 100},
                {"tool_id": "support", "num_tools": 50},
            ]
        })

        client = ContextCacheClient()
        result = client.list_tool_sets()
        assert len(result) == 2
        assert result[0]["tool_id"] == "merchant"

    @patch("context_cache.client.requests.Session")
    def test_remove_tools(self, MockSession):
        session = MockSession.return_value
        session.headers = {}
        session.delete.return_value = _mock_response({"status": "removed"})

        client = ContextCacheClient()
        result = client.remove_tools("merchant")
        assert result["status"] == "removed"
