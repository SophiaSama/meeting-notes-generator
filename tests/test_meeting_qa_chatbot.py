"""Seam tests for Meeting Q&A Chatbot (Ticket 01: Meeting Context Capture & Single-Turn Grounded Q&A)."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pytest

from meeting_notes.pipeline import MeetingContext, PipelineRunResult, PipelineService
from meeting_notes.ui import MeetingNotesApp

PROXY_ARGS = dict(
    base_url="http://localhost:4000",
    api_key="test-key",
    model="gpt-4o",
)


def _content_text_blob(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _all_messages_text(messages: list[dict]) -> str:
    return "\n".join(_content_text_blob(m.get("content", "")) for m in messages)


@pytest.mark.e2e
def test_run_pipeline_captures_serializable_meeting_context_and_preserves_4_tuple(
    monkeypatch, pipeline, synthetic_video
):
    monkeypatch.setenv("LLM_OCR_MODEL", "mistral-document-ai-2505")
    notes_reply = pipeline.llm_service.reply

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        prompt = (kwargs.get("system_prompt") or "").lower()
        if "ocr extraction assistant" in prompt:
            return "Slide 3: Q4 budget capped at $1.2M"
        return notes_reply

    pipeline.llm_service.call_llm = fake_call_llm

    result = pipeline.run_pipeline(
        str(synthetic_video),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
    )

    # 1. Existing 4-tuple unpacking continues to work unchanged.
    assert isinstance(result, PipelineRunResult)
    thumbnails, notes_text, html_notes, out_path = result
    assert len(thumbnails) == 2
    assert "Executive Summary" in notes_text
    assert html_notes.startswith("<!DOCTYPE html>")
    assert Path(out_path).exists()

    # 2. MeetingContext is attached and also retrievable via return_context=True.
    ctx = result.meeting_context
    assert isinstance(ctx, MeetingContext)
    assert isinstance(ctx, dict)

    # Must be 100% JSON-serializable (no raw PIL Image or base64 blobs).
    serialized = json.dumps(ctx)
    assert "base64_jpeg" not in serialized
    assert "pil_image" not in serialized

    assert ctx["info"]["filename"].startswith("team_sync_")
    assert ctx["output_language"] == "English"
    assert ctx["notes_style"] == "Executive Summary"
    assert ctx["transcript_source_label"] == "transcript"
    assert "welcome to the test meeting" in ctx["transcript_evidence"].lower()
    assert "The team agreed to ship the meeting-notes pipeline." in ctx["markdown_notes"]

    frames_meta = ctx["frames_metadata"]
    assert len(frames_meta) == 2
    assert frames_meta[0]["index"] == 1
    assert frames_meta[0]["time_str"] == "0:02"
    assert "Q4 budget capped at $1.2M" in frames_meta[0]["ocr_text"]
    assert "budget" in frames_meta[0]["ocr_keywords"]
    assert "pil_image" not in frames_meta[0]
    assert "base64_jpeg" not in frames_meta[0]


@pytest.mark.e2e
def test_generate_then_query_video_flow_grounds_on_notes_transcript_and_ocr(
    monkeypatch, pipeline, synthetic_video
):
    monkeypatch.setenv("LLM_OCR_MODEL", "mistral-document-ai-2505")
    notes_reply = pipeline.llm_service.reply

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        prompt = (kwargs.get("system_prompt") or "").lower()
        if "ocr extraction assistant" in prompt:
            return "Latency target: 42ms p99"
        if kwargs.get("stage") == "meeting_qa":
            return "According to Screenshot 1 at 0:02, the latency target is 42ms p99, and Alice owns the export format."
        return notes_reply

    pipeline.llm_service.call_llm = fake_call_llm

    _, _, _, _, meeting_context = pipeline.run_pipeline(
        str(synthetic_video),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="Spanish",
        return_context=True,
    )

    reply, updated_history = pipeline.query_meeting(
        "What was the latency target on the slide and who owns the export format?",
        [],
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )

    assert "42ms p99" in reply
    assert "Screenshot 1 at 0:02" in reply
    assert updated_history == [
        {
            "role": "user",
            "content": "What was the latency target on the slide and who owns the export format?",
        },
        {
            "role": "assistant",
            "content": "According to Screenshot 1 at 0:02, the latency target is 42ms p99, and Alice owns the export format.",
        },
    ]

    qa_call = pipeline.llm_service.calls[-1]
    assert qa_call["kwargs"]["stage"] == "meeting_qa"
    qa_system_prompt = qa_call["kwargs"]["system_prompt"]
    assert "Spanish" in qa_system_prompt
    assert "not mentioned" in qa_system_prompt.lower()
    assert "chain-of-thought" in qa_system_prompt.lower()

    qa_prompt_blob = _all_messages_text(qa_call["messages"])
    assert "welcome to the test meeting" in qa_prompt_blob.lower()
    assert "finalize the export format" in qa_prompt_blob
    assert "[Screenshot 1 at 0:02]" in qa_prompt_blob
    assert "Latency target: 42ms p99" in qa_prompt_blob
    assert "What was the latency target on the slide and who owns the export format?" in qa_prompt_blob


@pytest.mark.e2e
def test_generate_then_query_transcript_only_flow(pipeline, tmp_path):
    transcript_path = tmp_path / "standup.transcript.txt"
    transcript_path.write_text(
        "[00:03:15] Bob: We are postponing the database migration to next Tuesday due to schema lock risks.",
        encoding="utf-8",
    )

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        if kwargs.get("stage") == "meeting_qa":
            return "At [00:03:15], Bob stated the database migration is postponed to next Tuesday due to schema lock risks."
        return "## Executive Summary\nDatabase migration discussed."

    pipeline.llm_service.call_llm = fake_call_llm

    _, _, _, _, meeting_context = pipeline.run_pipeline_with_context(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Detailed Notes",
        output_language="English",
        transcript_file=str(transcript_path),
    )

    assert meeting_context["frames_metadata"] == []
    assert "postponing the database migration" in meeting_context["transcript_evidence"]

    reply, updated_history = pipeline.query_meeting_safe(
        "Why did Bob postpone the database migration?",
        [],
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="French",
    )

    assert "schema lock risks" in reply
    assert len(updated_history) == 2
    qa_call = pipeline.llm_service.calls[-1]
    assert "French" in qa_call["kwargs"]["system_prompt"]
    qa_blob = _all_messages_text(qa_call["messages"])
    assert "No screenshots captured (transcript-only mode)." in qa_blob
    assert "postponing the database migration to next Tuesday" in qa_blob


@pytest.mark.parametrize("missing_ctx", [None, {}, {"markdown_notes": "   ", "transcript_evidence": ""}])
def test_missing_context_guardrail_does_not_call_llm(pipeline, missing_ctx):
    with pytest.raises(gr.Error, match="generate meeting notes first"):
        pipeline.query_meeting(
            "What were the action items?",
            [],
            missing_ctx,
            PROXY_ARGS["base_url"],
            PROXY_ARGS["api_key"],
            PROXY_ARGS["model"],
        )

    assert pipeline.llm_service.calls == []

    reply, history = pipeline.query_meeting_safe(
        "What were the action items?",
        [],
        missing_ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "generate meeting notes first" in reply.lower()
    assert len(history) == 2
    assert history[-1]["role"] == "assistant"
    assert "generate meeting notes first" in history[-1]["content"].lower()
    assert pipeline.llm_service.calls == []


@pytest.mark.parametrize("blank_question", ["", "   ", "\n\t  "])
def test_empty_question_guardrail_leaves_history_unchanged_without_calling_llm(
    pipeline, blank_question
):
    meeting_context = PipelineService.build_meeting_context(
        info={"filename": "sync.mp4", "duration_str": "1:00", "resolution": "1280x720"},
        markdown_notes="## Summary\nShipped.",
        transcript_evidence="Hello everyone.",
        transcript_source_label="transcript",
        frames=[],
        output_language="English",
        notes_style="Executive Summary",
    )
    prior_history = [
        {"role": "user", "content": "Who attended?"},
        {"role": "assistant", "content": "Alice and Bob."},
    ]

    reply, updated_history = pipeline.query_meeting(
        blank_question,
        prior_history,
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert reply == ""
    assert updated_history == prior_history
    assert pipeline.llm_service.calls == []

    safe_reply, safe_history = pipeline.query_meeting_safe(
        blank_question,
        prior_history,
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert safe_reply == ""
    assert safe_history == prior_history
    assert pipeline.llm_service.calls == []


def test_ui_wiring_builds_chat_tab_and_state():
    app = MeetingNotesApp()
    assert isinstance(app.meeting_context_state, gr.State)
    assert isinstance(app.chatbot, gr.Chatbot)
    assert isinstance(app.chat_input, gr.Textbox)
    assert isinstance(app.send_btn, gr.Button)
    assert isinstance(app.clear_chat_btn, gr.Button)
