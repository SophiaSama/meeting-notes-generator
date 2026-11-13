"""Seam and UI lifecycle tests for Meeting Q&A Chatbot (Ticket 03: Chat Session Lifecycle Controls)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
import pytest

from meeting_notes.pipeline import PipelineService
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
def test_consecutive_generations_replace_context_and_clear_chat_with_zero_bleed(
    pipeline: PipelineService, tmp_path: Path
):
    """Running Generate Meeting Notes on Meeting A then Meeting B replaces meeting_context,
    clears prior chat history, and ensures subsequent queries only contain Meeting B evidence.
    """
    app = MeetingNotesApp(pipeline_service=pipeline)

    transcript_a = tmp_path / "meeting_alpha.transcript.txt"
    transcript_a.write_text(
        "[00:01:00] Alice: Project Phoenix launch date is October 14 with budget $500k.",
        encoding="utf-8",
    )
    transcript_b = tmp_path / "meeting_beta.transcript.txt"
    transcript_b.write_text(
        "[00:02:30] Carol: Project Nebula freeze is November 22 with budget $900k.",
        encoding="utf-8",
    )

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        blob = _all_messages_text(messages)
        if kwargs.get("stage") == "meeting_qa":
            if "Project Nebula" in blob:
                return "Project Nebula freezes on November 22 with a $900k budget."
            return "Project Phoenix launches on October 14 with a $500k budget."
        if "Project Phoenix" in blob:
            return "## Executive Summary\nAlpha notes: Project Phoenix launches October 14."
        return "## Executive Summary\nBeta notes: Project Nebula freezes November 22."

    pipeline.llm_service.call_llm = fake_call_llm

    # 1. Generate notes for Meeting A via UI callback.
    (
        _thumbs_a,
        notes_a,
        _html_a,
        _out_a,
        ctx_a,
        chat_hist_a,
        chat_in_a,
        empty_up_a,
        results_up_a,
        error_up_a,
        starter_up_a,
        _stage_progress_u,
    ) = app._on_generate_click(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_a),
    )

    assert "Project Phoenix" in notes_a
    assert ctx_a is not None
    assert "Project Phoenix" in ctx_a["markdown_notes"]
    assert "Project Phoenix" in ctx_a["transcript_evidence"]
    assert chat_hist_a == []
    assert chat_in_a == ""
    assert empty_up_a.get("visible") is False
    assert results_up_a.get("visible") is True
    assert error_up_a.get("visible") is False
    assert starter_up_a.get("visible") is True

    # Ask a question on Meeting A to populate chat history.
    chat_hist_after_q1, cleared_input = app._on_chat_submit(
        "When does Project Phoenix launch?",
        chat_hist_a,
        ctx_a,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="English",
    )
    assert cleared_input == ""
    assert len(chat_hist_after_q1) == 2
    assert "October 14" in chat_hist_after_q1[-1]["content"]

    # 2. Generate notes for Meeting B -> replaces context, clears chat history and input.
    (
        _thumbs_b,
        notes_b,
        _html_b,
        _out_b,
        ctx_b,
        chat_hist_b,
        chat_in_b,
        empty_up_b,
        results_up_b,
        error_up_b,
        starter_up_b,
        _stage_progress_u,
    ) = app._on_generate_click(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_b),
    )

    assert "Project Nebula" in notes_b
    assert ctx_b is not None
    assert ctx_b is not ctx_a
    assert chat_hist_b == []
    assert chat_in_b == ""
    assert empty_up_b.get("visible") is False
    assert results_up_b.get("visible") is True
    assert error_up_b.get("visible") is False
    assert starter_up_b.get("visible") is True
    assert "Project Nebula" in ctx_b["markdown_notes"]
    assert "Project Nebula" in ctx_b["transcript_evidence"]
    assert "Project Phoenix" not in ctx_b["markdown_notes"]
    assert "Project Phoenix" not in ctx_b["transcript_evidence"]

    # 3. Query against Meeting B's fresh context and cleared history.
    chat_hist_after_q2, _ = app._on_chat_submit(
        "What is the freeze date and budget?",
        chat_hist_b,
        ctx_b,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="English",
    )
    assert len(chat_hist_after_q2) == 2
    assert "November 22" in chat_hist_after_q2[-1]["content"]

    qa_call_b = pipeline.llm_service.calls[-1]
    assert qa_call_b["kwargs"]["stage"] == "meeting_qa"
    qa_blob_b = _all_messages_text(qa_call_b["messages"])
    assert "Project Nebula" in qa_blob_b
    assert "November 22" in qa_blob_b
    assert "Project Phoenix" not in qa_blob_b
    assert "October 14" not in qa_blob_b
    assert "$500k" not in qa_blob_b


@pytest.mark.e2e
def test_consecutive_video_runs_replace_frame_ocr_and_transcript_context(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: PipelineService,
    synthetic_video: Path,
    tmp_path: Path,
):
    """Running pipeline on two distinct video uploads replaces frames_metadata, transcript_evidence,
    and markdown_notes so subsequent queries only see the second video's OCR and transcript.
    """
    monkeypatch.setenv("LLM_OCR_MODEL", "mistral-document-ai-2505")
    app = MeetingNotesApp(pipeline_service=pipeline)

    video_a = tmp_path / "video_alpha.mp4"
    video_a.write_bytes(synthetic_video.read_bytes())
    video_b = tmp_path / "video_beta.mp4"
    video_b.write_bytes(synthetic_video.read_bytes())

    active_video = {"tag": "alpha"}

    def fake_transcribe(video_path, **kwargs):
        video = Path(video_path)
        transcripts_dir = Path.cwd() / "artifacts" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcripts_dir / f"{video.stem}.transcript.txt"
        transcript_plain_path = transcripts_dir / f"{video.stem}.full.txt"
        manifest_path = transcripts_dir / f"{video.stem}.manifest.json"
        if active_video["tag"] == "alpha":
            text = "[Chunk 1]\nSpeaker Alpha: The Alpha server port is 8081.\n"
        else:
            text = "[Chunk 1]\nSpeaker Beta: The Beta server port is 9092.\n"
        transcript_path.write_text(text, encoding="utf-8")
        transcript_plain_path.write_text(text, encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        return {
            "transcript_text": text,
            "transcript_path": str(transcript_path),
            "transcript_plain_path": str(transcript_plain_path),
            "manifest_path": str(manifest_path),
            "manifest": {"coverage_ratio": 1.0, "status": "ok", "actual_chunks": 1},
        }

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        prompt = (kwargs.get("system_prompt") or "").lower()
        if "ocr extraction assistant" in prompt:
            return (
                "Alpha Slide: SLA 99.5%"
                if active_video["tag"] == "alpha"
                else "Beta Slide: SLA 99.99%"
            )
        if kwargs.get("stage") == "meeting_qa":
            return "Beta port is 9092 and SLA is 99.99%."
        return (
            "## Executive Summary\nAlpha video summary.\n@@SCREENSHOT_1@@"
            if active_video["tag"] == "alpha"
            else "## Executive Summary\nBeta video summary.\n@@SCREENSHOT_1@@"
        )

    pipeline.video_service.transcribe_video_chunked = fake_transcribe
    pipeline.llm_service.call_llm = fake_call_llm

    active_video["tag"] = "alpha"
    _, _, _, _, ctx_a, _, _, _, _, _, _, _ = app._on_generate_click(
        str(video_a),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
    )
    assert "Alpha Slide: SLA 99.5%" in ctx_a["frames_metadata"][0]["ocr_text"]

    active_video["tag"] = "beta"
    _, _, _, _, ctx_b, chat_hist_b, _, _, _, _, _, _ = app._on_generate_click(
        str(video_b),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
    )
    assert chat_hist_b == []
    assert "Beta Slide: SLA 99.99%" in ctx_b["frames_metadata"][0]["ocr_text"]

    reply, _ = pipeline.query_meeting(
        "What is the server port and SLA on the slide?",
        chat_hist_b,
        ctx_b,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "9092" in reply
    qa_blob = _all_messages_text(pipeline.llm_service.calls[-1]["messages"])
    assert "Beta Slide: SLA 99.99%" in qa_blob
    assert "Beta server port is 9092" in qa_blob
    assert "Alpha Slide" not in qa_blob
    assert "8081" not in qa_blob


@pytest.mark.e2e
def test_clear_chat_resets_only_conversation_and_input_while_preserving_context(
    pipeline: PipelineService, tmp_path: Path
):
    """Clicking Clear Chat clears chatbot history and question input while keeping
    meeting_context intact and immediately usable for a fresh line of questioning.
    """
    app = MeetingNotesApp(pipeline_service=pipeline)

    transcript_path = tmp_path / "design_review.transcript.txt"
    transcript_path.write_text(
        "[00:04:10] Dana: We selected PostgreSQL 16 for the billing service.",
        encoding="utf-8",
    )

    def fake_call_llm(messages, **kwargs):
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        if kwargs.get("stage") == "meeting_qa":
            return f"Grounded answer ({len(messages)} messages in payload): PostgreSQL 16."
        return "## Executive Summary\nSelected PostgreSQL 16 for billing."

    pipeline.llm_service.call_llm = fake_call_llm

    (
        thumbnails,
        notes_text,
        html_notes,
        out_path,
        meeting_context,
        chat_history,
        chat_input,
        empty_update,
        results_update,
        error_update,
        starter_update,
        stage_progress_update,
    ) = app._on_generate_click(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_path),
    )
    assert empty_update.get("visible") is False
    assert results_update.get("visible") is True
    assert error_update.get("visible") is False
    assert starter_update.get("visible") is True

    # First turn builds history.
    chat_history, chat_input = app._on_chat_submit(
        "Why did we discuss databases?",
        chat_history,
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="English",
    )
    assert len(chat_history) == 2

    # Clear Chat resets only chatbot and chat_input.
    cleared_history, cleared_input = app._on_clear_chat()
    assert cleared_history == []
    assert cleared_input == ""

    # Generated artifacts and meeting_context remain untouched and usable.
    assert Path(out_path).exists()
    assert "PostgreSQL 16" in notes_text
    assert html_notes.startswith("<!DOCTYPE html>")
    assert thumbnails == []
    assert meeting_context["info"]["filename"] == "design_review.transcript.txt"

    # Ask a new question after Clear Chat: LLM receives NO prior conversation turns,
    # only the grounding context message + current user question (2 messages total).
    fresh_history, fresh_input = app._on_chat_submit(
        "Which database version did Dana pick?",
        cleared_history,
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="English",
    )
    assert fresh_input == ""
    assert len(fresh_history) == 2
    assert fresh_history[0]["content"] == "Which database version did Dana pick?"

    last_qa_call = pipeline.llm_service.calls[-1]
    assert len(last_qa_call["messages"]) == 2
    sent_blob = _all_messages_text(last_qa_call["messages"])
    assert "Why did we discuss databases?" not in sent_blob
    assert "PostgreSQL 16" in sent_blob


@pytest.mark.e2e
def test_global_reset_clears_context_and_blocks_subsequent_queries_without_llm_call(
    pipeline: PipelineService, tmp_path: Path
):
    """Clicking Reset clears meeting_context, chatbot history, and question input alongside
    all other outputs, and subsequent chat queries trigger the missing-context guardrail.
    """
    app = MeetingNotesApp(pipeline_service=pipeline)

    transcript_path = tmp_path / "sprint_retro.transcript.txt"
    transcript_path.write_text(
        "[00:05:00] Eli: Deployment automation reduced rollbacks by 60%.",
        encoding="utf-8",
    )

    _, _, _, _, meeting_context, chat_history, _, _, _, _, _, _ = app._on_generate_click(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_path),
    )
    assert meeting_context is not None

    chat_history, _ = app._on_chat_submit(
        "How much did rollbacks drop?",
        chat_history,
        meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="English",
    )
    assert len(chat_history) == 2
    calls_before_reset = len(pipeline.llm_service.calls)

    # Global Reset clears all 15 outputs including meeting_context, chatbot, and chat_input.
    reset_outputs = app._on_reset()
    assert len(reset_outputs) == 16
    (
        video_val,
        transcript_val,
        gallery_val,
        notes_val,
        html_val,
        download_val,
        dup_banner_val,
        capture_hint_val,
        reset_meeting_context,
        reset_chatbot,
        reset_chat_input,
        reset_empty_state,
        reset_results,
        reset_error_banner,
        reset_starter_row,
        reset_stage_progress,
    ) = reset_outputs

    assert video_val is None
    assert transcript_val is None
    assert gallery_val == []
    assert notes_val == ""
    assert html_val == ""
    assert download_val is None
    assert dup_banner_val.get("visible") is False
    assert capture_hint_val.get("visible") is False
    assert reset_meeting_context is None
    assert reset_chatbot == []
    assert reset_chat_input == ""
    assert reset_empty_state.get("visible") is True
    assert reset_results.get("visible") is False
    assert reset_error_banner.get("visible") is False
    assert reset_error_banner.get("value") == ""
    assert reset_starter_row.get("visible") is False

    # Attempting to query after Reset triggers the missing-context guardrail without calling the LLM.
    with pytest.raises(gr.Error, match="generate meeting notes first"):
        pipeline.query_meeting(
            "What did Eli say?",
            reset_chatbot,
            reset_meeting_context,
            PROXY_ARGS["base_url"],
            PROXY_ARGS["api_key"],
            PROXY_ARGS["model"],
        )

    post_reset_history, post_reset_input = app._on_chat_submit(
        "What did Eli say?",
        reset_chatbot,
        reset_meeting_context,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        output_language="English",
    )
    assert post_reset_input == ""
    assert len(post_reset_history) == 2
    assert "generate meeting notes first" in post_reset_history[-1]["content"].lower()
    assert len(pipeline.llm_service.calls) == calls_before_reset


@pytest.mark.e2e
def test_failed_subsequent_generation_clears_stale_meeting_context_and_chat_history(
    pipeline: PipelineService, tmp_path: Path
):
    """If Meeting A succeeds and a subsequent Meeting B generation fails, _on_generate_click
    resets meeting_context to None and clears chatbot history so Meeting A does not stay active.
    """
    app = MeetingNotesApp(pipeline_service=pipeline)

    transcript_a = tmp_path / "meeting_one.transcript.txt"
    transcript_a.write_text(
        "[00:01:00] Frank: Secret codename is AURORA.",
        encoding="utf-8",
    )
    transcript_b = tmp_path / "meeting_two.transcript.txt"
    transcript_b.write_text(
        "[00:01:00] Grace: Second meeting content.",
        encoding="utf-8",
    )

    _, _, _, _, ctx_a, chat_hist_a, _, _, _, _, _, _ = app._on_generate_click(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_a),
    )
    assert ctx_a is not None

    # Populate chat history on Meeting A.
    chat_hist_a, _ = app._on_chat_submit(
        "What is the codename?",
        chat_hist_a,
        ctx_a,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert len(chat_hist_a) == 2

    # Now make the LLM fail during Meeting B's note generation.
    def failing_llm(*args, **kwargs):
        raise RuntimeError("upstream gateway timeout")

    pipeline.llm_service.call_llm = failing_llm

    (
        thumbnails_b,
        notes_b,
        html_b,
        out_b,
        ctx_b,
        chat_hist_b,
        chat_in_b,
        empty_up_b,
        results_up_b,
        error_up_b,
        starter_up_b,
        _stage_progress_u,
    ) = app._on_generate_click(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_b),
    )

    assert thumbnails_b == []
    assert notes_b == ""
    assert html_b == ""
    assert out_b is None
    assert ctx_b is None
    assert chat_hist_b == []
    assert chat_in_b == ""
    assert empty_up_b.get("visible") is True
    assert results_up_b.get("visible") is False
    assert error_up_b.get("visible") is True
    assert "error-banner" in error_up_b.get("value", "")
    assert starter_up_b.get("visible") is False

    # Subsequent chat submission with the resulting ctx_b must trigger the missing-context guardrail.
    guardrail_history, _ = app._on_chat_submit(
        "What is the codename?",
        chat_hist_b,
        ctx_b,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert len(guardrail_history) == 2
    assert "generate meeting notes first" in guardrail_history[-1]["content"].lower()


def test_concurrent_sessions_are_isolated_and_hold_no_instance_state(pipeline: PipelineService):
    """Two independent browser sessions querying through the same MeetingNotesApp /
    PipelineService instance do not bleed context or history into each other, and
    neither class stores active notes or chat history on self.
    """
    app = MeetingNotesApp(pipeline_service=pipeline)

    ctx_session_1 = PipelineService.build_meeting_context(
        info={"filename": "session1_roadmap.mp4", "duration_str": "15:00", "resolution": "1920x1080"},
        markdown_notes="## Summary\nSession 1 agreed to ship Feature Red.",
        transcript_evidence="[00:01:00] Alice: Feature Red is our top Q1 priority.",
        transcript_source_label="transcript",
        frames=[{"time_str": "0:10", "ocr_text": "Slide: Feature Red KPI 99.9%", "ocr_keywords": ["feature", "red"]}],
        output_language="English",
        notes_style="Executive Summary",
    )
    ctx_session_2 = PipelineService.build_meeting_context(
        info={"filename": "session2_finance.mp4", "duration_str": "20:00", "resolution": "1280x720"},
        markdown_notes="## Summary\nSession 2 approved Vendor Blue contract.",
        transcript_evidence="[00:02:00] Bob: Vendor Blue costs $45,000 annually.",
        transcript_source_label="transcript",
        frames=[{"time_str": "0:20", "ocr_text": "Slide: Vendor Blue $45k/yr", "ocr_keywords": ["vendor", "blue"]}],
        output_language="Spanish",
        notes_style="Action-Oriented",
    )

    history_session_1 = [
        {"role": "user", "content": "Who spoke in session 1?"},
        {"role": "assistant", "content": "Alice spoke about Feature Red."},
    ]
    history_session_2 = [
        {"role": "user", "content": "Who spoke in session 2?"},
        {"role": "assistant", "content": "Bob spoke about Vendor Blue."},
    ]

    recorded_payloads: dict[str, dict] = {}

    def fake_call_llm(messages, **kwargs):
        blob = _all_messages_text(messages)
        if "Feature Red" in blob:
            recorded_payloads["session_1"] = {"messages": messages, "kwargs": kwargs, "blob": blob}
            return "Session 1 answer: Feature Red KPI is 99.9%."
        if "Vendor Blue" in blob:
            recorded_payloads["session_2"] = {"messages": messages, "kwargs": kwargs, "blob": blob}
            return "Session 2 answer: Vendor Blue costs $45,000."
        raise AssertionError("Unexpected prompt payload")

    pipeline.llm_service.call_llm = fake_call_llm

    def _run_session_1():
        return app._on_chat_submit(
            "What is the KPI for Feature Red?",
            history_session_1,
            ctx_session_1,
            PROXY_ARGS["base_url"],
            PROXY_ARGS["api_key"],
            PROXY_ARGS["model"],
            output_language="English",
            reasoning_effort="low",
        )

    def _run_session_2():
        return app._on_chat_submit(
            "How much does Vendor Blue cost?",
            history_session_2,
            ctx_session_2,
            PROXY_ARGS["base_url"],
            PROXY_ARGS["api_key"],
            PROXY_ARGS["model"],
            output_language="Spanish",
            reasoning_effort="high",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_run_session_1)
        f2 = executor.submit(_run_session_2)
        updated_hist_1, input_1 = f1.result()
        updated_hist_2, input_2 = f2.result()

    assert input_1 == ""
    assert input_2 == ""
    assert len(updated_hist_1) == 4
    assert len(updated_hist_2) == 4
    assert "Feature Red KPI is 99.9%" in updated_hist_1[-1]["content"]
    assert "Vendor Blue costs $45,000" in updated_hist_2[-1]["content"]

    # Verify zero cross-session bleed in prompts sent to the LLM.
    blob_1 = recorded_payloads["session_1"]["blob"]
    blob_2 = recorded_payloads["session_2"]["blob"]
    assert "Feature Red" in blob_1
    assert "Alice spoke about Feature Red" in blob_1
    assert "Vendor Blue" not in blob_1
    assert "Bob spoke about Vendor Blue" not in blob_1

    assert "Vendor Blue" in blob_2
    assert "Bob spoke about Vendor Blue" in blob_2
    assert "Feature Red" not in blob_2
    assert "Alice spoke about Feature Red" not in blob_2

    # Verify neither PipelineService nor MeetingNotesApp stores active meeting context or chat history on self.
    assert not hasattr(pipeline, "meeting_context")
    assert not hasattr(pipeline, "chat_history")
    assert not hasattr(app, "meeting_context")
    assert not hasattr(app, "chat_history")
    assert app.meeting_context_state.value is None


def test_ui_lifecycle_event_wiring_preserves_notes_on_clear_chat_and_resets_all_on_reset():
    """Verify Gradio event graph wiring for Generate, Clear Chat, and Reset buttons."""
    app = MeetingNotesApp()

    # Chatbot visually indicates when no notes have been generated yet via placeholder.
    assert "generate meeting notes first" in (app.chatbot.placeholder or "").lower()
    assert app.meeting_context_state.value is None

    def _find_block_fn(method_name: str):
        for fn_obj in app.demo.fns.values():
            fn = getattr(fn_obj, "fn", None)
            if fn is not None and getattr(fn, "__name__", "") == method_name:
                return fn_obj
        raise AssertionError(f"Could not find Gradio function for {method_name}")

    clear_fn = _find_block_fn("_on_clear_chat")
    assert clear_fn.inputs == []
    assert [c._id for c in clear_fn.outputs] == [app.chatbot._id, app.chat_input._id]
    for untouched in (
        app.meeting_context_state._id,
        app.notes_output._id,
        app.html_preview._id,
        app.screenshots_gallery._id,
        app.download_file._id,
    ):
        assert untouched not in [c._id for c in clear_fn.outputs]

    gen_fn = _find_block_fn("_on_generate_click")
    assert [c._id for c in gen_fn.outputs] == [
        app.screenshots_gallery._id,
        app.notes_output._id,
        app.html_preview._id,
        app.download_file._id,
        app.meeting_context_state._id,
        app.chatbot._id,
        app.chat_input._id,
        app.empty_state_container._id,
        app.results_container._id,
        app.error_banner._id,
        app.starter_row._id,
        app.stage_progress_html._id,
    ]

    reset_fn = _find_block_fn("_on_reset")
    assert [c._id for c in reset_fn.outputs] == [
        app.video_input._id,
        app.transcript_input._id,
        app.screenshots_gallery._id,
        app.notes_output._id,
        app.html_preview._id,
        app.download_file._id,
        app.dup_banner._id,
        app.capture_hint._id,
        app.meeting_context_state._id,
        app.chatbot._id,
        app.chat_input._id,
        app.empty_state_container._id,
        app.results_container._id,
        app.error_banner._id,
        app.starter_row._id,
        app.stage_progress_html._id,
    ]
