"""Tests for reasoning-model handling in ``LLMService.call_llm``.

Covers the optional reasoning-control ``extra_body``, its graceful fallback
when the provider rejects it (e.g. always-thinking models such as Kimi K3),
model-aware request timeouts, and safe diagnostics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError

import meeting_notes.llm as llm_module
from meeting_notes.llm import LLMService

CALL_ARGS = dict(
    base_url="http://localhost:4000",
    api_key="k",
)


def _response(content: str = "ok", reasoning: str | None = None) -> SimpleNamespace:
    raw = {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": reasoning},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model_dump=lambda: raw,
    )


def _bad_request(message: str = "unsupported parameter: thinking") -> BadRequestError:
    request = httpx.Request("POST", "http://localhost:4000/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=None)


class _FakeOpenAI:
    """Records chat.completions.create kwargs; behaviour set per-test."""

    instances: list["_FakeOpenAI"] = []
    side_effects: list = []
    requests: list[dict] = []
    http_timeouts: list = []
    default_headers: list = []

    def __init__(self, *, base_url, api_key, http_client=None, default_headers=None):
        _FakeOpenAI.default_headers.append(default_headers)
        if http_client is not None:
            _FakeOpenAI.http_timeouts.append(http_client.timeout)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _FakeOpenAI.requests.append(kwargs)
        effect = _FakeOpenAI.side_effects.pop(0) if _FakeOpenAI.side_effects else _response()
        if isinstance(effect, Exception):
            raise effect
        return effect


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeOpenAI.side_effects = []
    _FakeOpenAI.requests = []
    _FakeOpenAI.http_timeouts = []
    _FakeOpenAI.default_headers = []
    monkeypatch.setattr(llm_module, "OpenAI", _FakeOpenAI)
    for name in (
        "LLM_REASONING_CONTROL_EXTRA_BODY",
        "LLM_REASONING_MODEL_PREFIXES",
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "LLM_EXTRA_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)
    return _FakeOpenAI


def _call(service: LLMService, model: str) -> str:
    return service.call_llm(
        [{"role": "user", "content": "hi"}], model=model, max_tokens=100, **CALL_ARGS
    )


def test_no_extra_body_sent_by_default(fake_openai):
    assert _call(LLMService(), "Kimi-K2.6") == "ok"
    assert "extra_body" not in fake_openai.requests[0]


def test_no_custom_headers_sent_by_default(fake_openai):
    _call(LLMService(), "gpt-4o")
    assert fake_openai.default_headers[0] is None


def test_extra_headers_from_env_are_sent(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_EXTRA_HEADERS", '{"HTTP-Referer": "https://example.com", "X-Title": "notes"}')
    _call(LLMService(), "gpt-4o")
    assert fake_openai.default_headers[0] == {"HTTP-Referer": "https://example.com", "X-Title": "notes"}


@pytest.mark.parametrize("raw", ["{not json", '["a", "b"]'])
def test_invalid_extra_headers_are_ignored(monkeypatch, fake_openai, raw):
    monkeypatch.setenv("LLM_EXTRA_HEADERS", raw)
    _call(LLMService(), "gpt-4o")
    assert fake_openai.default_headers[0] is None


def test_reasoning_control_sent_only_for_reasoning_models(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_REASONING_CONTROL_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    service = LLMService()

    _call(service, "Kimi-K2.6")
    _call(service, "gpt-4o")

    assert fake_openai.requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in fake_openai.requests[1]


def test_rejected_reasoning_control_is_dropped_and_remembered(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_REASONING_CONTROL_EXTRA_BODY", '{"reasoning_effort": "low"}')
    fake_openai.side_effects = [_bad_request(), _response("first"), _response("second")]
    service = LLMService()

    assert _call(service, "Kimi-K3") == "first"
    assert _call(service, "Kimi-K3") == "second"

    assert "extra_body" in fake_openai.requests[0]
    assert "extra_body" not in fake_openai.requests[1]
    # Remembered for the process: the next call does not try again.
    assert "extra_body" not in fake_openai.requests[2]
    assert len(fake_openai.requests) == 3


def test_bad_request_without_reasoning_control_propagates(fake_openai):
    fake_openai.side_effects = [_bad_request("invalid messages")]
    with pytest.raises(BadRequestError):
        _call(LLMService(), "Kimi-K2.6")
    assert len(fake_openai.requests) == 1


def test_invalid_reasoning_control_json_is_ignored(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_REASONING_CONTROL_EXTRA_BODY", "{not json")
    _call(LLMService(), "Kimi-K2.6")
    assert "extra_body" not in fake_openai.requests[0]


def test_reasoning_models_get_longer_timeout(fake_openai):
    service = LLMService()
    _call(service, "Kimi-K2.6")
    _call(service, "gpt-4o")
    assert fake_openai.http_timeouts[0].read == LLMService.DEFAULT_REASONING_REQUEST_TIMEOUT_SECONDS
    assert fake_openai.http_timeouts[1].read == LLMService.DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_timeout_env_override(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "450")
    _call(LLMService(), "gpt-4o")
    assert fake_openai.http_timeouts[0].read == 450.0


def test_custom_reasoning_model_prefixes(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_MODEL_PREFIXES", "my-thinker, other")
    assert LLMService._is_reasoning_model("My-Thinker-v2")
    assert not LLMService._is_reasoning_model("Kimi-K2.6")


# ---------------------------------------------------------------------------
# UI / MCP reasoning effort ({"reasoning": {"effort": ...}})
# ---------------------------------------------------------------------------


def _unprocessable(message: str = "unknown field: reasoning"):
    from openai import UnprocessableEntityError

    request = httpx.Request("POST", "http://localhost:4000/chat/completions")
    response = httpx.Response(422, request=request)
    return UnprocessableEntityError(message, response=response, body=None)


def test_effort_sent_to_every_model(fake_openai):
    service = LLMService()
    with service.reasoning_effort("low"):
        _call(service, "gpt-4o")
        _call(service, "Kimi-K2.6")
    assert fake_openai.requests[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert fake_openai.requests[1]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_effort_none_sends_nothing_and_overrides_env(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_REASONING_CONTROL_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    service = LLMService()
    with service.reasoning_effort("none"):
        _call(service, "Kimi-K2.6")
    assert "extra_body" not in fake_openai.requests[0]


def test_effort_replaces_env_payload(monkeypatch, fake_openai):
    monkeypatch.setenv("LLM_REASONING_CONTROL_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    service = LLMService()
    with service.reasoning_effort("high"):
        _call(service, "Kimi-K2.6")
    assert fake_openai.requests[0]["extra_body"] == {"reasoning": {"effort": "high"}}


def test_effort_scope_is_restored_after_block(fake_openai):
    service = LLMService()
    with service.reasoning_effort("medium"):
        pass
    _call(service, "gpt-4o")
    assert "extra_body" not in fake_openai.requests[0]
    assert LLMService.current_reasoning_effort() is None


@pytest.mark.parametrize("error_factory", [_bad_request, _unprocessable])
def test_rejected_effort_is_silently_dropped_and_remembered(fake_openai, caplog, error_factory):
    fake_openai.side_effects = [error_factory(), _response("first"), _response("second")]
    service = LLMService()
    with caplog.at_level("INFO"), service.reasoning_effort("low"):
        assert _call(service, "gpt-4o") == "first"
        assert _call(service, "gpt-4o") == "second"

    assert fake_openai.requests[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "extra_body" not in fake_openai.requests[1]
    assert "extra_body" not in fake_openai.requests[2]
    assert len(fake_openai.requests) == 3
    assert "WARNING" not in caplog.text


def test_rejection_is_per_model(fake_openai):
    fake_openai.side_effects = [_bad_request(), _response("plain"), _response("other")]
    service = LLMService()
    with service.reasoning_effort("low"):
        _call(service, "gpt-4o")
        _call(service, "Kimi-K2.6")
    assert fake_openai.requests[2]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_rejection_is_per_effort_level(fake_openai):
    fake_openai.side_effects = [_bad_request(), _response(), _response()]
    service = LLMService()
    with service.reasoning_effort("low"):
        _call(service, "gpt-4o")
    with service.reasoning_effort("high"):
        _call(service, "gpt-4o")
    assert fake_openai.requests[2]["extra_body"] == {"reasoning": {"effort": "high"}}


def test_unrelated_bad_request_is_not_remembered(fake_openai):
    # Both with and without the setting fail -> error propagates, setting kept.
    fake_openai.side_effects = [_bad_request("prompt too long"), _bad_request("prompt too long")]
    service = LLMService()
    with service.reasoning_effort("low"):
        with pytest.raises(BadRequestError):
            _call(service, "gpt-4o")
        _call(service, "gpt-4o")
    assert fake_openai.requests[2]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_test_connection_passes_effort(fake_openai):
    result = LLMService().test_connection(
        "http://localhost:4000", "k", "gpt-4o", "u1", "u2", "app", "minimal"
    )
    assert result.startswith("✅")
    assert fake_openai.requests[0]["extra_body"] == {"reasoning": {"effort": "minimal"}}


def test_batch_reruns_without_rejected_effort(monkeypatch, fake_openai):
    from meeting_notes.llm import _BatchRequestsFailedError

    service = LLMService()
    seen: list = []

    def fake_once(*, reasoning_control, **kwargs):
        seen.append(reasoning_control)
        if reasoning_control:
            raise _BatchRequestsFailedError("mapreq-000001: status_code=400")
        return ["ok"]

    monkeypatch.setattr(service, "_call_llm_batch_chat_once", fake_once)
    batch_args: dict[str, Any] = dict(
        messages_list=[[{"role": "user", "content": "hi"}]], model="gpt-4o", **CALL_ARGS
    )
    with service.reasoning_effort("low"):
        assert service.call_llm_batch_chat(**batch_args) == ["ok"]
        assert service.call_llm_batch_chat(**batch_args) == ["ok"]
    assert seen == [{"reasoning": {"effort": "low"}}, None, None]


def test_diagnostics_log_reasoning_length_not_content(fake_openai, caplog):
    secret = "SECRET-CHAIN-OF-THOUGHT"
    fake_openai.side_effects = [_response("answer", reasoning=secret)]
    with caplog.at_level("INFO"):
        _call(LLMService(), "Kimi-K2.6")
    assert f"reasoning_chars={len(secret)}" in caplog.text
    assert secret not in caplog.text
