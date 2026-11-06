"""Seam tests for Ticket 02: Multi-Turn Conversation History, Request Budgeting & Reasoning/Output-Limit Resilience."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import meeting_notes.llm as llm_module
from meeting_notes.llm import LLMOutputLimitError, LLMResponse, LLMService
from meeting_notes.notes import NotesService
from meeting_notes.pipeline import PipelineService
from meeting_notes.video import VideoService

PROXY_ARGS = dict(
    base_url="http://localhost:4000",
    api_key="test-key",
    model="gpt-4o",
)


def _sample_meeting_context(
    *,
    markdown_notes: str = "## Summary\n- Alice owns the API rollout.\n- Bob raised concerns about migration locks.",
    transcript_evidence: str = "[00:01:10] Bob: I am worried about database locks.\n[00:02:30] Alice: We will run it off-hours.",
    transcript_source_label: str = "transcript",
    output_language: str = "English",
):
    return PipelineService.build_meeting_context(
        info={
            "filename": "architecture_sync.mp4",
            "duration": 180.0,
            "duration_str": "3:00",
            "size_mb": "4.50",
            "resolution": "1920x1080",
        },
        markdown_notes=markdown_notes,
        transcript_evidence=transcript_evidence,
        transcript_source_label=transcript_source_label,
        frames=[
            {
                "time_str": "0:45",
                "ocr_text": "Rollout Window: Sunday 02:00 UTC",
                "ocr_keywords": ["rollout", "window", "sunday"],
            }
        ],
        output_language=output_language,
        notes_style="Detailed Notes",
    )


def test_multi_turn_history_preservation_and_input_normalization(pipeline):
    ctx = _sample_meeting_context()
    replies = iter(
        [
            "Bob raised concerns about database locks at [00:01:10].",
            "Alice responded at [00:02:30] that the migration will run off-hours.",
            "Screenshot 1 at 0:45 specifies Sunday 02:00 UTC.",
        ]
    )

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        return next(replies)

    pipeline.llm_service.call_llm = fake_call_llm

    # Turn 1 (empty history)
    r1, h1 = pipeline.query_meeting(
        "What concern did Bob raise?",
        [],
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "database locks" in r1
    assert len(h1) == 2

    # Turn 2 (pass tuple history format + multimodal list content to verify normalization)
    legacy_and_multimodal_history = [
        ("What concern did Bob raise?", "Bob raised concerns about database locks at [00:01:10]."),
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who replied to him?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Alice replied."}],
        },
    ]
    r2, h2 = pipeline.query_meeting(
        "How did Alice address it?",
        legacy_and_multimodal_history,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "off-hours" in r2
    assert len(h2) == 6

    # Turn 3 (pass standard dict history from Turn 2)
    r3, h3 = pipeline.query_meeting(
        "What exact window was on the slide?",
        h2,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "Sunday 02:00 UTC" in r3
    assert len(h3) == 8

    # Verify chronological order in messages sent to LLM on Turn 3
    sent_messages = pipeline.llm_service.calls[-1]["messages"]
    # Index 0 is the grounding message, followed by the 6 prior turn messages, then current question
    assert len(sent_messages) == 8
    assert "Meeting Context" in sent_messages[0]["content"]
    assert sent_messages[1] == {"role": "user", "content": "What concern did Bob raise?"}
    assert sent_messages[2] == {
        "role": "assistant",
        "content": "Bob raised concerns about database locks at [00:01:10].",
    }
    assert sent_messages[3] == {"role": "user", "content": "Who replied to him?"}
    assert sent_messages[4] == {"role": "assistant", "content": "Alice replied."}
    assert sent_messages[5] == {"role": "user", "content": "How did Alice address it?"}
    assert sent_messages[6] == {
        "role": "assistant",
        "content": "Alice responded at [00:02:30] that the migration will run off-hours.",
    }
    assert sent_messages[7] == {
        "role": "user",
        "content": "What exact window was on the slide?",
    }


def test_request_budget_trims_oldest_turns_and_bounds_transcript(monkeypatch, pipeline):
    long_transcript = "TRANSCRIPT_START " + ("Detailed spoken segment. " * 250) + " TRANSCRIPT_END"
    ctx = _sample_meeting_context(transcript_evidence=long_transcript)

    prior_history = [
        {"role": "user", "content": "OLDEST_TURN_Q1: " + ("x" * 600)},
        {"role": "assistant", "content": "OLDEST_TURN_A1: " + ("y" * 600)},
        {"role": "user", "content": "MIDDLE_TURN_Q2: " + ("x" * 600)},
        {"role": "assistant", "content": "MIDDLE_TURN_A2: " + ("y" * 600)},
        {"role": "user", "content": "RECENT_TURN_Q3: What is the latest status?"},
        {"role": "assistant", "content": "RECENT_TURN_A3: Ready for Sunday rollout."},
    ]

    # Set a tight byte limit so full history + long transcript cannot fit without trimming
    monkeypatch.setenv("LLM_MAX_CHAT_REQUEST_BYTES", "3600")

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        return "Trimmed response ok."

    pipeline.llm_service.call_llm = fake_call_llm

    reply, updated_history = pipeline.query_meeting(
        "Confirm the rollout day?",
        prior_history,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )

    assert reply == "Trimmed response ok."
    # Returned UI chat history still retains the full conversation + new turn
    assert len(updated_history) == len(prior_history) + 2

    qa_call = pipeline.llm_service.calls[-1]
    sent_messages = qa_call["messages"]
    estimated_bytes = PipelineService._estimate_chat_request_bytes(
        sent_messages,
        model=PROXY_ARGS["model"],
        max_tokens=qa_call["kwargs"]["max_tokens"],
        system_prompt=qa_call["kwargs"]["system_prompt"],
    )
    assert estimated_bytes <= PipelineService._chat_request_budget_bytes()

    sent_blob = "\n".join(m["content"] for m in sent_messages)
    # Oldest turn was trimmed first to fit budget
    assert "OLDEST_TURN_Q1" not in sent_blob
    assert "OLDEST_TURN_A1" not in sent_blob
    # Grounding context and current question remain intact
    assert "Meeting Context" in sent_messages[0]["content"]
    assert "TRANSCRIPT_START" in sent_messages[0]["content"]
    assert sent_messages[-1] == {"role": "user", "content": "Confirm the rollout day?"}


@pytest.mark.e2e
def test_map_reduce_condensed_transcript_digest_captured_and_used_in_qa(
    monkeypatch, pipeline, tmp_path
):
    monkeypatch.setenv("LLM_NOTES_MAP_REDUCE_TRIGGER_CHARS", "300")
    monkeypatch.setenv("LLM_NOTES_MAP_REDUCE_SEGMENT_CHARS", "500")

    long_tx_path = tmp_path / "marathon_meeting.txt"
    long_tx_path.write_text(
        "[Chunk 1]\n"
        + ("Alice discussed the Q1 infrastructure budget and Kubernetes node autoscaling. " * 6)
        + "\n[Chunk 2]\n"
        + ("Bob confirmed the security audit passed with zero critical findings on Friday. " * 6),
        encoding="utf-8",
    )

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        stage = kwargs.get("stage")
        if stage == "segment_summary":
            return "### Segment\n- Topics discussed: Autoscaling and security audit."
        if stage == "reduce_summary":
            return "## Condensed Transcript Digest\n- Autoscaling approved.\n- Security audit passed with 0 criticals."
        if stage == "final_notes":
            return "## Executive Summary\nInfrastructure and security reviewed."
        if stage == "meeting_qa":
            return "The security audit had zero critical findings."
        return "ok"

    pipeline.llm_service.call_llm = fake_call_llm

    _, _, _, _, meeting_context = pipeline.run_pipeline_with_context(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(long_tx_path),
    )

    assert (
        meeting_context["transcript_source_label"]
        == "condensed transcript digest generated from all transcript segments"
    )
    assert "Condensed Transcript Digest" in meeting_context["transcript_evidence"]
    assert "Security audit passed with 0 criticals" in meeting_context["transcript_evidence"]

    reply, _ = pipeline.query_meeting(
        "How many critical findings were in the security audit?",
        [],
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "zero critical findings" in reply
    qa_messages = pipeline.llm_service.calls[-1]["messages"]
    assert "condensed transcript digest generated from all transcript segments" in qa_messages[0]["content"]
    assert "Security audit passed with 0 criticals" in qa_messages[0]["content"]


def test_dynamic_model_and_reasoning_effort_switching_and_session_isolation(pipeline):
    ctx = _sample_meeting_context()
    observed_efforts: list[tuple[str, str | None, int]] = []

    def fake_call_llm(messages, **kwargs):
        model = kwargs["model"]
        max_tokens = kwargs["max_tokens"]
        effort = LLMService.current_reasoning_effort()
        observed_efforts.append((model, effort, max_tokens))
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs, "effort": effort})
        return f"Answered by {model} (effort={effort})"

    pipeline.llm_service.call_llm = fake_call_llm

    # Turn 1: standard model, effort="low"
    r1, h1 = pipeline.query_meeting(
        "Turn 1 question?",
        [],
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        "gpt-4o",
        reasoning_effort="low",
    )
    assert r1 == "Answered by gpt-4o (effort=low)"

    # Turn 2: switch to reasoning model "Kimi-K2.6" and effort="high"
    r2, h2 = pipeline.query_meeting(
        "Turn 2 complex question?",
        h1,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        "Kimi-K2.6",
        reasoning_effort="high",
    )
    assert r2 == "Answered by Kimi-K2.6 (effort=high)"

    assert observed_efforts[0] == ("gpt-4o", "low", PipelineService.DEFAULT_QA_MAX_TOKENS)
    assert observed_efforts[1] == (
        "Kimi-K2.6",
        "high",
        PipelineService.DEFAULT_REASONING_QA_MAX_TOKENS,
    )
    # ContextVar is clean outside the call
    assert LLMService.current_reasoning_effort() is None

    # Verify concurrent session isolation via ThreadPoolExecutor
    def _run_concurrent_turn(effort_val: str) -> str:
        reply, _ = pipeline.query_meeting(
            f"Question with {effort_val}",
            [],
            ctx,
            PROXY_ARGS["base_url"],
            PROXY_ARGS["api_key"],
            "Kimi-K2.6",
            reasoning_effort=effort_val,
        )
        return reply

    with ThreadPoolExecutor(max_workers=4) as pool:
        efforts = ["none", "minimal", "low", "high"]
        results = list(pool.map(_run_concurrent_turn, efforts))

    assert results == [
        "Answered by Kimi-K2.6 (effort=none)",
        "Answered by Kimi-K2.6 (effort=minimal)",
        "Answered by Kimi-K2.6 (effort=low)",
        "Answered by Kimi-K2.6 (effort=high)",
    ]


def test_reasoning_model_output_limit_retry_and_fallback_and_cot_stripping(monkeypatch):
    ctx = _sample_meeting_context()
    fallback_model = "fallback-from-env"
    monkeypatch.setenv("LLM_NOTES_FALLBACK_MODEL", fallback_model)

    # Wire a real LLMService with a fake OpenAI client to verify both:
    # 1. reasoning_content (chain-of-thought) is stripped and never leaked into chat reply
    # 2. LLMOutputLimitError triggers token-multiplier retry and fallback model recovery on stage="meeting_qa"
    requests_seen: list[dict] = []
    side_effects: list = []

    class _StubOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            requests_seen.append(kwargs)
            effect = side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect

    def _make_sdk_response(content: str, reasoning: str | None = None, finish_reason: str = "stop"):
        raw = {
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": reasoning},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 500,
                "completion_tokens_details": {"reasoning_tokens": 450} if reasoning else None,
            },
        }
        msg = SimpleNamespace(content=content, reasoning_content=reasoning)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
            model_dump=lambda: raw,
        )

    monkeypatch.setattr(llm_module, "OpenAI", _StubOpenAI)
    llm = LLMService()
    service = PipelineService(llm, VideoService(llm), NotesService())

    # Scenario A: First attempt exhausts output budget on thinking (empty content + finish_reason="length"),
    # second attempt (retry with larger token budget) succeeds and includes reasoning_content that must be stripped.
    secret_cot = "INTERNAL_CHAIN_OF_THOUGHT_DO_NOT_SHOW_IN_CHAT"
    side_effects.extend(
        [
            _make_sdk_response("", reasoning="Spent all tokens thinking...", finish_reason="length"),
            _make_sdk_response(
                "Final visible answer: Rollout is Sunday at 02:00 UTC.",
                reasoning=secret_cot,
                finish_reason="stop",
            ),
        ]
    )

    reply, history = service.query_meeting(
        "When is the rollout window?",
        [],
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        "Kimi-K2.6",
        reasoning_effort="medium",
    )

    assert reply == "Final visible answer: Rollout is Sunday at 02:00 UTC."
    assert secret_cot not in reply
    assert secret_cot not in history[-1]["content"]
    assert len(requests_seen) == 2
    assert requests_seen[0]["model"] == "Kimi-K2.6"
    assert requests_seen[0]["max_tokens"] == PipelineService.DEFAULT_REASONING_QA_MAX_TOKENS
    assert requests_seen[0]["extra_body"] == {"reasoning": {"effort": "medium"}}
    assert requests_seen[1]["model"] == "Kimi-K2.6"
    assert requests_seen[1]["max_tokens"] > requests_seen[0]["max_tokens"]

    # Scenario B: Both primary attempts hit output limit -> recovers via fallback model ("gpt-4o")
    requests_seen.clear()
    side_effects.extend(
        [
            _make_sdk_response("", reasoning="Still thinking 1...", finish_reason="length"),
            _make_sdk_response("", reasoning="Still thinking 2...", finish_reason="length"),
            _make_sdk_response("Fallback model answer: Alice owns the rollout.", finish_reason="stop"),
        ]
    )

    fallback_reply, fallback_history = service.query_meeting(
        "Who owns the rollout?",
        history,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        "Kimi-K2.6",
    )
    assert fallback_reply == "Fallback model answer: Alice owns the rollout."
    assert len(fallback_history) == 4
    assert len(requests_seen) == 3
    assert requests_seen[0]["model"] == "Kimi-K2.6"
    assert requests_seen[1]["model"] == "Kimi-K2.6"
    assert requests_seen[2]["model"] == fallback_model


def test_query_meeting_safe_surfaces_friendly_inline_error_and_preserves_context(pipeline):
    ctx = _sample_meeting_context()
    original_notes = ctx["markdown_notes"]
    prior_history = [
        {"role": "user", "content": "Who is leading the rollout?"},
        {"role": "assistant", "content": "Alice is leading the rollout."},
    ]

    def broken_call_llm(messages, **kwargs):
        raise ConnectionError("Connection refused by upstream proxy at localhost:4000")

    pipeline.llm_service.call_llm = broken_call_llm

    reply, updated_history = pipeline.query_meeting_safe(
        "What happens if the migration locks?",
        prior_history,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )

    assert "Unable to answer the question right now" in reply
    assert "ConnectionError" in reply
    # Prior history turns are preserved and the failed turn is appended inline
    assert len(updated_history) == 4
    assert updated_history[:2] == prior_history
    assert updated_history[2] == {
        "role": "user",
        "content": "What happens if the migration locks?",
    }
    assert updated_history[3] == {"role": "assistant", "content": reply}
    # Meeting context and generated notes remain untouched
    assert ctx["markdown_notes"] == original_notes


def test_fallback_model_comes_only_from_env(monkeypatch):
    """No hardcoded fallback: the value comes from LLM_NOTES_FALLBACK_MODEL (.env)."""
    monkeypatch.delenv("LLM_NOTES_FALLBACK_MODEL", raising=False)
    assert PipelineService._fallback_model() == ""
    monkeypatch.setenv("LLM_NOTES_FALLBACK_MODEL", "  gemini-3.5-flash  ")
    assert PipelineService._fallback_model() == "gemini-3.5-flash"
