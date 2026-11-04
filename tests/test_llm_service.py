from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from meeting_notes.llm import LLMOutputLimitError, LLMService


def _chat_completion_response(
    *,
    content: str | None,
    finish_reason: str = "stop",
    usage: dict | None = None,
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    raw = {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model_dump=lambda: raw,
    )


class _FakeFilesApi:
    def __init__(self, output_jsonl: str):
        self.output_jsonl = output_jsonl
        self.create_calls: list[dict] = []

    def create(self, *, file, purpose: str):
        self.create_calls.append(
            {
                "file_name": getattr(file, "name", ""),
                "purpose": purpose,
            }
        )
        return SimpleNamespace(id="file-input-1")

    def content(self, file_id: str):
        assert file_id == "file-output-1"
        return SimpleNamespace(text=self.output_jsonl)


class _FakeBatchesApi:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.retrieve_calls = 0

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
    ):
        self.create_calls.append(
            {
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
            }
        )
        return SimpleNamespace(id="batch-1")

    def retrieve(self, batch_id: str):
        assert batch_id == "batch-1"
        self.retrieve_calls += 1
        return SimpleNamespace(status="completed", output_file_id="file-output-1")


class _FakeClient:
    def __init__(self, output_jsonl: str):
        self.files = _FakeFilesApi(output_jsonl)
        self.batches = _FakeBatchesApi()


def _batch_output_line(custom_id: str, content: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {"content": content},
                            "finish_reason": "stop",
                        }
                    ]
                },
            },
        }
    )


def test_call_llm_batch_chat_returns_results_in_input_order(monkeypatch: pytest.MonkeyPatch):
    service = LLMService()
    output_jsonl = "\n".join(
        [
            _batch_output_line("mapreq-000002", "segment-two"),
            _batch_output_line("mapreq-000001", "segment-one"),
        ]
    )
    fake_client = _FakeClient(output_jsonl)
    monkeypatch.setattr(service, "build_client", lambda base_url, api_key: fake_client)
    monkeypatch.setenv("LLM_BATCH_MAX_WAIT_SECONDS", "10")

    results = service.call_llm_batch_chat(
        messages_list=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        base_url="http://localhost:4000",
        api_key="k",
        model="gpt-4o",
        max_tokens=123,
        system_prompt="sys",
    )

    assert results == ["segment-one", "segment-two"]
    assert fake_client.files.create_calls[0]["purpose"] == "batch"
    assert fake_client.batches.create_calls[0]["endpoint"] == "/v1/chat/completions"


def test_call_llm_batch_chat_rejects_jsonl_above_soft_limit(monkeypatch: pytest.MonkeyPatch):
    service = LLMService()
    monkeypatch.setattr(service, "_batch_jsonl_soft_limit_bytes", lambda: 250)
    monkeypatch.setattr(
        service,
        "build_client",
        lambda base_url, api_key: (_ for _ in ()).throw(AssertionError("must not call network")),
    )

    with pytest.raises(RuntimeError, match="safety threshold"):
        service.call_llm_batch_chat(
            messages_list=[[{"role": "user", "content": "x" * 10_000}]],
            base_url="http://localhost:4000",
            api_key="k",
            model="gpt-4o",
        )


def test_call_llm_batch_chat_raises_on_batch_item_error(monkeypatch: pytest.MonkeyPatch):
    service = LLMService()
    output_jsonl = json.dumps(
        {
            "custom_id": "mapreq-000001",
            "error": {"message": "upstream failure"},
        }
    )
    fake_client = _FakeClient(output_jsonl)
    monkeypatch.setattr(service, "build_client", lambda base_url, api_key: fake_client)

    with pytest.raises(RuntimeError, match="failed requests"):
        service.call_llm_batch_chat(
            messages_list=[[{"role": "user", "content": "first"}]],
            base_url="http://localhost:4000",
            api_key="k",
            model="gpt-4o",
        )


def test_build_llm_response_handles_null_completion_token_details():
    service = LLMService()
    response = _chat_completion_response(
        content="Visible answer",
        usage={
            "prompt_tokens": 2629,
            "completion_tokens": 25,
            "completion_tokens_details": None,
        },
    )

    result = service._build_llm_response(response)

    assert result.content == "Visible answer"
    assert result.input_tokens == 2629
    assert result.output_tokens == 25
    assert result.reasoning_tokens is None


def test_build_llm_response_handles_null_usage():
    service = LLMService()
    response = _chat_completion_response(content="Visible answer", usage=None)

    result = service._build_llm_response(response)

    assert result.content == "Visible answer"
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.reasoning_tokens is None


def test_validate_kimi_length_response_reports_reasoning_without_returning_it():
    service = LLMService()
    response = _chat_completion_response(
        content=None,
        reasoning_content="Internal analysis only",
        finish_reason="length",
        usage={
            "prompt_tokens": 2629,
            "completion_tokens": 2200,
            "completion_tokens_details": None,
        },
    )
    normalized = service._build_llm_response(response)

    with pytest.raises(LLMOutputLimitError) as exc_info:
        service._validate_llm_response(
            normalized,
            "Kimi-K2.6",
            requested_max_tokens=2200,
            stage="final_notes",
        )

    error = exc_info.value
    assert error.requested_max_tokens == 2200
    assert error.finish_reason == "length"
    assert error.input_tokens == 2629
    assert error.output_tokens == 2200
    assert error.has_separate_reasoning is True
    assert "Internal analysis only" not in str(error)


def test_length_finish_with_visible_content_is_returned():
    service = LLMService()
    normalized = service._build_llm_response(
        _chat_completion_response(content="Partial visible answer", finish_reason="length")
    )

    assert service._validate_llm_response(normalized, "Kimi-K2.6") == "Partial visible answer"


def test_safe_diagnostics_do_not_log_reasoning_content(caplog: pytest.LogCaptureFixture):
    service = LLMService()
    normalized = service._build_llm_response(
        _chat_completion_response(
            content=None,
            reasoning_content="SECRET CHAIN OF THOUGHT",
            finish_reason="length",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
    )

    with caplog.at_level("INFO"):
        service._log_call_diagnostics(
            llm_response=normalized,
            model="Kimi-K2.6",
            requested_max_tokens=20,
            stage="segment_summary",
            attempt=1,
        )

    assert "model=Kimi-K2.6" in caplog.text
    assert "requested_max_tokens=20" in caplog.text
    assert "has_separate_reasoning=True" in caplog.text
    assert "SECRET CHAIN OF THOUGHT" not in caplog.text


def test_extract_text_skips_reasoning_and_thinking_parts():
    """_extract_text_from_message should ignore content parts with
    type='reasoning' or type='thinking' (chain-of-thought leakage)."""

    service = LLMService()

    # When content is a list of structured parts
    message = {
        "content": [
            {"type": "reasoning", "text": "The user wants formal meeting minutes..."},
            {"type": "thinking", "text": "I should structure the output..."},
            {"type": "text", "text": "# Meeting Minutes\n\n## Attendees\n- Alice\n- Bob"},
            {
                "type": "text",
                "text": "## Agenda\n1. Project update",
                "annotations": [],
            },
        ]
    }

    result = service._extract_text_from_message(message)

    # Only the plain "text" type parts should appear
    assert "The user wants" not in result
    assert "I should structure" not in result
    assert "Meeting Minutes" in result
    assert "Attendees" in result
    assert "Project update" in result
    assert "Alice" in result
    assert "Bob" in result


def test_extract_text_skips_text_parts_with_reasoning_annotation():
    """_extract_text_from_message should skip text parts that have
    annotations marking them as reasoning/thinking."""

    service = LLMService()

    message = {
        "content": [
            {
                "type": "text",
                "text": "Chain-of-thought leaked here",
                "annotations": [{"type": "reasoning"}],
            },
            {
                "type": "text",
                "text": "Another leaked thought",
                "annotations": [{"type": "thinking"}],
            },
            {
                "type": "text",
                "text": "## Action Items\n- Item 1\n- Item 2",
            },
        ]
    }

    result = service._extract_text_from_message(message)

    assert "Chain-of-thought leaked" not in result
    assert "Another leaked thought" not in result
    assert "Action Items" in result
    assert "Item 1" in result
    assert "Item 2" in result


def test_extract_text_handles_plain_string_content():
    """_extract_text_from_message should handle plain string content (backward compatibility)."""

    service = LLMService()

    result = service._extract_text_from_message({"content": "Simple plain text response"})

    assert result == "Simple plain text response"


def test_extract_text_handles_empty():
    """_extract_text_from_message should return empty string for empty/missing content."""

    service = LLMService()

    assert service._extract_text_from_message({}) == ""
    assert service._extract_text_from_message({"content": ""}) == ""
    assert service._extract_text_from_message({"content": []}) == ""