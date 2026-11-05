"""End-to-end tests for the meeting-notes generation pipeline.

These drive ``PipelineService.run_pipeline`` the same way the Gradio UI does,
using a real synthetic video and the real notes/markdown/HTML rendering. Only
the network/ffmpeg boundaries are faked (see ``conftest.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

from meeting_notes.llm import LLMOutputLimitError, LLMResponse
from meeting_notes.pipeline import PipelineService

PROXY_ARGS = dict(
    base_url="http://localhost:4000",
    api_key="test-key",
    model="gpt-4o",
)


def _run(pipeline, video_path: Path):
    return pipeline.run_pipeline(
        str(video_path),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
    )


def _content_text_blob(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


@pytest.mark.e2e
def test_pipeline_returns_thumbnails_notes_and_html(pipeline, synthetic_video):
    thumbnails, notes_text, html_notes, html_path = _run(pipeline, synthetic_video)

    # Two frames are expected from a 6s video sampled every 2s.
    assert len(thumbnails) == 2
    assert "Executive Summary" in notes_text
    assert html_notes.startswith("<!DOCTYPE html>")
    assert Path(html_path).exists()


@pytest.mark.e2e
def test_pipeline_appends_artifact_summary_to_notes(pipeline, synthetic_video):
    _, notes_text, _, _ = _run(pipeline, synthetic_video)

    assert "Artifacts" in notes_text
    assert "Minutes markdown:" in notes_text
    assert "Transcript:" in notes_text
    assert "Screenshots:" in notes_text
    assert "interval" in notes_text
    assert "max" in notes_text


@pytest.mark.e2e
def test_pipeline_writes_minutes_markdown_with_screenshots(pipeline, synthetic_video):
    _run(pipeline, synthetic_video)

    minutes_dir = Path.cwd() / "artifacts" / "meeting_minutes"
    minutes_files = list(minutes_dir.glob("*.minutes.md"))
    assert len(minutes_files) == 1

    doc = minutes_files[0].read_text(encoding="utf-8")
    assert doc.startswith("# Meeting Minutes")
    # The upload is staged with a timestamp suffix (team_sync_<ts>.mp4).
    assert "source_video: team_sync_" in doc
    assert "screenshot_interval_seconds:" in doc
    assert "max_screenshots:" in doc
    assert "screenshots_captured:" in doc
    # Placeholders must be replaced with markdown image links.
    assert "@@SCREENSHOT_" not in doc
    assert "![Screenshot at" in doc

    saved_images = list((minutes_dir / "images").glob("team_sync_*/screenshot_*.jpg"))
    assert len(saved_images) == 2


@pytest.mark.e2e
def test_pipeline_embeds_screenshots_in_html(pipeline, synthetic_video):
    _, _, html_notes, _ = _run(pipeline, synthetic_video)

    assert "@@SCREENSHOT_" not in html_notes
    assert html_notes.count('<figure class="screenshot">') == 2
    assert "data:image/jpeg;base64," in html_notes


@pytest.mark.e2e
def test_pipeline_writes_transcript_artifacts(pipeline, synthetic_video):
    _run(pipeline, synthetic_video)

    transcripts_dir = Path.cwd() / "artifacts" / "transcripts"
    # Artifacts are named after the staged (timestamped) video stem.
    assert len(list(transcripts_dir.glob("team_sync_*.transcript.txt"))) == 1
    assert len(list(transcripts_dir.glob("team_sync_*.full.txt"))) == 1
    assert len(list(transcripts_dir.glob("team_sync_*.manifest.json"))) == 1


@pytest.mark.e2e
def test_pipeline_passes_transcript_to_llm(pipeline, synthetic_video):
    _run(pipeline, synthetic_video)

    assert len(pipeline.llm_service.calls) >= 1
    notes_call = next(
        call
        for call in pipeline.llm_service.calls
        if "meeting notes assistant" in call["kwargs"].get("system_prompt", "").lower()
    )
    user_message = notes_call["messages"][0]["content"]
    text_blob = _content_text_blob(user_message)
    assert "welcome to the test meeting" in text_blob.lower()
    system_prompt = notes_call["kwargs"]["system_prompt"]
    assert "English" in system_prompt


@pytest.mark.e2e
def test_pipeline_uses_text_only_when_multimodal_payload_is_too_large(
    monkeypatch, pipeline, synthetic_video
):
    huge_image = "A" * 6_000_000
    frames = [
        {
            "time_seconds": 2,
            "time_str": "0:02",
            "pil_image": Image.new("RGB", (16, 16), "white"),
            "base64_jpeg": huge_image,
            "ocr_text": "",
            "ocr_keywords": [],
        },
        {
            "time_seconds": 4,
            "time_str": "0:04",
            "pil_image": Image.new("RGB", (16, 16), "white"),
            "base64_jpeg": huge_image,
            "ocr_text": "",
            "ocr_keywords": [],
        },
    ]
    monkeypatch.setattr(pipeline.video_service, "extract_frames", lambda *args: frames)
    monkeypatch.setattr(
        pipeline,
        "_enrich_frames_with_document_model_ocr",
        lambda frames, **kwargs: {"available": False, "engine": "none"},
    )
    pipeline.llm_service.reply = "## Executive Summary\nCovered."

    _run(pipeline, synthetic_video)

    notes_call = next(
        call
        for call in pipeline.llm_service.calls
        if "meeting notes assistant" in call["kwargs"].get("system_prompt", "").lower()
    )
    content = notes_call["messages"][0]["content"]
    assert isinstance(content, str)
    assert "Screenshot 1 at 0:02" in content


@pytest.mark.e2e
def test_capture_recommendation_caps_long_video_to_request_budget(monkeypatch):
    monkeypatch.delenv("LLM_MAX_CHAT_REQUEST_BYTES", raising=False)

    recommendation = PipelineService.recommend_capture_settings(
        duration_seconds=3600,
        resolution="1920x1080",
        avg_frame_request_bytes=200_000,
    )

    assert recommendation["max_frames"] == 42
    assert recommendation["interval_seconds"] == 80
    assert recommendation["estimated_request_bytes"] <= recommendation["budget_bytes"]


@pytest.mark.e2e
def test_capture_recommendation_keeps_one_per_minute_when_budget_allows(monkeypatch):
    monkeypatch.delenv("LLM_MAX_CHAT_REQUEST_BYTES", raising=False)

    recommendation = PipelineService.recommend_capture_settings(
        duration_seconds=900,
        resolution="1280x720",
        avg_frame_request_bytes=100_000,
    )

    assert recommendation["max_frames"] == 15
    assert recommendation["interval_seconds"] == 55
    assert recommendation["estimated_request_bytes"] <= recommendation["budget_bytes"]


@pytest.mark.e2e
def test_pipeline_uses_document_model_ocr(monkeypatch, pipeline, synthetic_video):
    monkeypatch.setenv("LLM_OCR_MODEL", "mistral-document-ai-2505")

    captured_calls = []
    notes_reply = pipeline.llm_service.reply

    def fake_call_llm(messages, **kwargs):
        captured_calls.append({"messages": messages, "kwargs": kwargs})
        prompt = (kwargs.get("system_prompt") or "").lower()
        if "ocr extraction assistant" in prompt:
            return "Dashboard speed 42"
        return notes_reply

    pipeline.llm_service.call_llm = fake_call_llm

    _, notes_text, _, _ = _run(pipeline, synthetic_video)

    ocr_calls = [
        call for call in captured_calls
        if "ocr extraction assistant" in call["kwargs"].get("system_prompt", "").lower()
    ]
    assert len(ocr_calls) == 2
    assert all(call["kwargs"]["model"] == "mistral-document-ai-2505" for call in ocr_calls)

    notes_call = next(
        call for call in captured_calls
        if "meeting notes assistant" in call["kwargs"].get("system_prompt", "").lower()
    )
    notes_prompt_text = _content_text_blob(notes_call["messages"][0]["content"])
    assert "ocr hint: dashboard speed 42" in notes_prompt_text.lower()
    assert "- OCR available: True" in notes_text


@pytest.mark.e2e
def test_pipeline_falls_back_when_document_ocr_fails(monkeypatch, pipeline, synthetic_video):
    monkeypatch.setenv("LLM_OCR_MODEL", "mistral-document-ai-2505")

    captured_calls = []
    notes_reply = pipeline.llm_service.reply

    def fake_call_llm(messages, **kwargs):
        captured_calls.append({"messages": messages, "kwargs": kwargs})
        prompt = (kwargs.get("system_prompt") or "").lower()
        if "ocr extraction assistant" in prompt:
            raise RuntimeError("document OCR failed")
        return notes_reply

    def fake_local_ocr(frames):
        for frame in frames:
            frame["ocr_text"] = "fallback local text"
            frame["ocr_keywords"] = ["fallback", "local", "text"]
        return {"available": True, "engine": "local-test"}

    pipeline.llm_service.call_llm = fake_call_llm
    pipeline.video_service.enrich_frames_with_ocr = fake_local_ocr

    _, notes_text, _, _ = _run(pipeline, synthetic_video)

    ocr_calls = [
        call for call in captured_calls
        if "ocr extraction assistant" in call["kwargs"].get("system_prompt", "").lower()
    ]
    assert len(ocr_calls) == 2

    notes_call = next(
        call for call in captured_calls
        if "meeting notes assistant" in call["kwargs"].get("system_prompt", "").lower()
    )
    notes_prompt_text = _content_text_blob(notes_call["messages"][0]["content"])
    assert "ocr hint: fallback local text" in notes_prompt_text.lower()
    assert "- OCR available: True" in notes_text


@pytest.mark.e2e
def test_pipeline_ocr_disabled_by_default(monkeypatch, pipeline, synthetic_video):
    monkeypatch.delenv("LLM_OCR_MODEL", raising=False)
    _, notes_text, _, _ = _run(pipeline, synthetic_video)
    assert "- OCR available: False (disabled by default)" in notes_text


@pytest.mark.e2e
def test_pipeline_rejects_missing_video(pipeline):
    with pytest.raises(gr.Error):
        pipeline.run_pipeline(
            None,
            PROXY_ARGS["base_url"], PROXY_ARGS["api_key"], PROXY_ARGS["model"],
            interval_seconds=2, max_frames=8,
            notes_style="Executive Summary", output_language="English",
        )


@pytest.mark.e2e
def test_pipeline_rejects_blank_base_url(pipeline, synthetic_video):
    with pytest.raises(gr.Error):
        pipeline.run_pipeline(
            str(synthetic_video),
            "   ", PROXY_ARGS["api_key"], PROXY_ARGS["model"],
            interval_seconds=2, max_frames=8,
            notes_style="Executive Summary", output_language="English",
        )


@pytest.mark.e2e
def test_run_pipeline_safe_returns_friendly_error_on_failure(pipeline, synthetic_video):
    # Force the underlying generation to fail; the safe wrapper must not raise.
    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    pipeline.llm_service.call_llm = boom

    thumbnails, notes_text, html_notes, html_path = pipeline.run_pipeline_safe(
        str(synthetic_video),
        PROXY_ARGS["base_url"], PROXY_ARGS["api_key"], PROXY_ARGS["model"],
        interval_seconds=2, max_frames=8,
        notes_style="Executive Summary", output_language="English",
    )

    assert thumbnails == []
    assert "Generation failed" in notes_text
    assert "RuntimeError: model exploded" in notes_text
    assert "Generation failed" in html_notes
    assert html_path is None


@pytest.mark.e2e
def test_pipeline_allows_regeneration_for_duplicate(pipeline, synthetic_video, tmp_path):
    # First upload processes normally and records the upload filename.
    _run(pipeline, synthetic_video)

    # Re-uploading the same filename (Gradio hands us a fresh temp path) is
    # still re-generated: generation is never blocked, it just reuses the
    # existing staged copy.
    reupload_dir = tmp_path / "reupload"
    reupload_dir.mkdir()
    duplicate_video = reupload_dir / synthetic_video.name
    duplicate_video.write_bytes(synthetic_video.read_bytes())

    thumbnails, notes_text, html_notes, html_path = _run(pipeline, duplicate_video)

    # Notes are produced again rather than skipped with a duplicate warning.
    assert len(thumbnails) == 2
    assert "already uploaded" not in notes_text.lower()
    assert "Artifacts" in notes_text
    assert html_notes.startswith("<!DOCTYPE html>")
    assert Path(html_path).exists()


@pytest.mark.e2e
@pytest.mark.filterwarnings(
    "ignore:Duplicate video detected:UserWarning"
)
def test_check_upload_for_duplicate_informs_at_upload_time(pipeline, synthetic_video, tmp_path):
    # No banner before the file has ever been processed.
    update = pipeline.check_upload_for_duplicate(str(synthetic_video))
    assert update["visible"] is False

    # Process it once so the upload filename is recorded.
    _run(pipeline, synthetic_video)

    # Re-uploading the same filename (fresh temp path) shows a visible duplicate
    # banner, but does NOT reject/clear the upload.
    reupload_dir = tmp_path / "reupload"
    reupload_dir.mkdir()
    duplicate_video = reupload_dir / synthetic_video.name
    duplicate_video.write_bytes(synthetic_video.read_bytes())
    update = pipeline.check_upload_for_duplicate(str(duplicate_video))
    assert update["visible"] is True
    assert "duplicate video detected" in update["value"].lower()
    assert "existing notes: artifacts/meeting_minutes/" in update["value"].lower()
    assert "still generate" in update["value"].lower()


@pytest.mark.e2e
def test_check_upload_for_duplicate_ignores_none_input(pipeline):
    # No file present -> banner stays hidden.
    update = pipeline.check_upload_for_duplicate(None)
    assert update["visible"] is False
    assert update["value"] == ""


@pytest.mark.e2e
@pytest.mark.filterwarnings(
    "ignore:Duplicate video detected:UserWarning"
)
def test_check_upload_for_duplicate_falls_back_to_minutes_file_for_legacy_index(pipeline, synthetic_video, tmp_path):
    # First run creates a staged video and minutes markdown.
    _run(pipeline, synthetic_video)

    index_path = Path.cwd() / "artifacts" / "uploads" / ".upload_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index[synthetic_video.name]
    entry.pop("minutes_name", None)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    reupload_dir = tmp_path / "reupload"
    reupload_dir.mkdir()
    duplicate_video = reupload_dir / synthetic_video.name
    duplicate_video.write_bytes(synthetic_video.read_bytes())

    update = pipeline.check_upload_for_duplicate(str(duplicate_video))
    assert update["visible"] is True
    assert "existing notes: artifacts/meeting_minutes/" in update["value"].lower()
    assert "team_sync_" in update["value"].lower()
    assert ".minutes.md" in update["value"].lower()


@pytest.mark.e2e
def test_pipeline_reuses_staged_copy_for_duplicate_content(pipeline, synthetic_video, tmp_path):
    uploads_dir = Path.cwd() / "artifacts" / "uploads"

    # First run stages exactly one copy of the video.
    _run(pipeline, synthetic_video)
    staged_after_first = [p for p in uploads_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    assert len(staged_after_first) == 1

    # Re-generating from the same upload filename reuses that staged copy
    # instead of writing a second one.
    reupload_dir = tmp_path / "reupload"
    reupload_dir.mkdir()
    duplicate_video = reupload_dir / synthetic_video.name
    duplicate_video.write_bytes(synthetic_video.read_bytes())
    _run(pipeline, duplicate_video)

    staged_after_duplicate = [p for p in uploads_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    assert len(staged_after_duplicate) == 1
    assert staged_after_duplicate[0].name == staged_after_first[0].name


@pytest.mark.e2e
def test_pipeline_allows_distinct_videos(pipeline, synthetic_video, tmp_path):
    # A different upload filename must not be treated as a duplicate.
    _run(pipeline, synthetic_video)

    other_video = tmp_path / "other.mp4"
    other_video.write_bytes(synthetic_video.read_bytes())

    # Should run the full pipeline without raising a duplicate error.
    thumbnails, notes_text, _, _ = _run(pipeline, other_video)
    assert "Artifacts" in notes_text


@pytest.mark.e2e
def test_pipeline_supports_user_uploaded_transcript_only(pipeline, tmp_path):
    transcript_path = tmp_path / "meeting.transcript.txt"
    transcript_path.write_text(
        "Project update: release moved to Friday. Action: Bob publishes release notes.",
        encoding="utf-8",
    )

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("video processing should be skipped in transcript-only mode")

    pipeline.video_service.get_video_info = should_not_be_called
    pipeline.video_service.extract_frames = should_not_be_called
    pipeline.video_service.transcribe_video_chunked = should_not_be_called

    thumbnails, notes_text, html_notes, html_path = pipeline.run_pipeline(
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

    assert thumbnails == []
    assert "transcript-only mode was used" in notes_text
    assert html_notes.startswith("<!DOCTYPE html>")
    assert Path(html_path).exists()


@pytest.mark.e2e
def test_pipeline_uses_map_reduce_for_long_transcripts(monkeypatch, pipeline, tmp_path):
    long_transcript = "\n\n".join(
        [
            "[Chunk 1]\nKickoff discussion and goals.",
            "[Chunk 2]\nDecision: Use segmented transcript summarization.",
            "[Chunk 3]\nAction: Alice prepares mapping by Friday.",
            "[Chunk 4]\nWrap-up: Thanks everyone, keep in touch.",
        ]
    )
    transcript_path = tmp_path / "long_meeting.transcript.txt"
    transcript_path.write_text(long_transcript, encoding="utf-8")

    monkeypatch.setenv("LLM_NOTES_MAP_REDUCE_TRIGGER_CHARS", "120")

    captured_calls: list[dict] = []
    segment_counter = {"n": 0}

    def fake_call_llm(messages, **kwargs):
        system_prompt = (kwargs.get("system_prompt") or "").lower()
        captured_calls.append({"messages": messages, "kwargs": kwargs})
        if "transcript segment analyst" in system_prompt:
            segment_counter["n"] += 1
            return f"### Segment {segment_counter['n']}\n- Topics discussed: Segment {segment_counter['n']}"
        if "meeting transcript synthesizer" in system_prompt:
            return (
                "## Condensed Transcript Digest\n"
                "### Chronological Flow\n"
                "- Kickoff\n- Decision\n- Action\n- Wrap-up\n"
                "### Decisions\n- Use segmented transcript summarization.\n"
                "### Action Items (owner -> task)\n- Alice -> prepares mapping by Friday.\n"
                "### Open Questions / Risks\n- None noted.\n"
                "### Coverage Notes\n- Full transcript covered."
            )
        return "## Executive Summary\nMap-reduce final notes."

    pipeline.llm_service.call_llm = fake_call_llm

    _, notes_text, _, _ = pipeline.run_pipeline(
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

    assert any(
        "transcript segment analyst"
        in (call["kwargs"].get("system_prompt") or "").lower()
        for call in captured_calls
    )
    assert any(
        "meeting transcript synthesizer"
        in (call["kwargs"].get("system_prompt") or "").lower()
        for call in captured_calls
    )

    notes_call = next(
        call
        for call in captured_calls
        if "meeting notes assistant" in (call["kwargs"].get("system_prompt") or "").lower()
    )
    notes_prompt_text = _content_text_blob(notes_call["messages"][0]["content"])
    assert "condensed transcript digest generated from all transcript segments" in notes_prompt_text.lower()
    assert "long-transcript map-reduce summarization was applied" in notes_text


def _output_limit_error(*, stage: str, max_tokens: int) -> LLMOutputLimitError:
    return LLMOutputLimitError(
        model="Kimi-K2.6",
        requested_max_tokens=max_tokens,
        response=LLMResponse(content="", finish_reason="length", output_tokens=max_tokens),
        stage=stage,
    )


def test_length_retry_is_bounded_and_uses_larger_budget(monkeypatch, pipeline):
    monkeypatch.setenv("LLM_NOTES_LENGTH_RETRY_TOKEN_MULTIPLIER", "2")
    calls: list[dict] = []

    def fake_call(messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _output_limit_error(stage="final_notes", max_tokens=1000)
        return "visible notes"

    pipeline.llm_service.call_llm = fake_call
    result = pipeline._call_llm_with_length_retry(
        [{"role": "user", "content": "meeting"}],
        stage="final_notes",
        max_tokens=1000,
        base_url="http://localhost:4000",
        api_key="k",
        model="Kimi-K2.6",
    )

    assert result == "visible notes"
    assert [call["max_tokens"] for call in calls] == [1000, 2000]
    assert [call["attempt"] for call in calls] == [1, 2]


def test_length_retry_never_attempts_third_call(monkeypatch, pipeline):
    monkeypatch.setenv("LLM_NOTES_FALLBACK_MODEL", "")
    calls = 0

    def always_limited(messages, **kwargs):
        nonlocal calls
        calls += 1
        raise _output_limit_error(
            stage=kwargs["stage"], max_tokens=kwargs["max_tokens"]
        )

    pipeline.llm_service.call_llm = always_limited
    with pytest.raises(LLMOutputLimitError):
        pipeline._call_llm_with_length_retry(
            [{"role": "user", "content": "meeting"}],
            stage="final_notes",
            max_tokens=1000,
            base_url="http://localhost:4000",
            api_key="k",
            model="Kimi-K2.6",
        )

    assert calls == 2


def test_non_length_failure_is_not_retried(pipeline):
    calls = 0

    def fail(messages, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("upstream unavailable")

    pipeline.llm_service.call_llm = fail
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        pipeline._call_llm_with_length_retry(
            [{"role": "user", "content": "meeting"}],
            stage="final_notes",
            max_tokens=1000,
            base_url="http://localhost:4000",
            api_key="k",
            model="Kimi-K2.6",
        )

    assert calls == 1


def test_stage_token_budgets_are_independently_configurable(monkeypatch):
    monkeypatch.setenv("LLM_NOTES_SEGMENT_MAX_TOKENS", "1111")
    monkeypatch.setenv("LLM_NOTES_REDUCE_MAX_TOKENS", "2222")
    monkeypatch.setenv("LLM_NOTES_FINAL_MAX_TOKENS", "3333")

    assert PipelineService._segment_max_tokens() == 1111
    assert PipelineService._reduce_max_tokens() == 2222
    assert PipelineService._final_max_tokens() == 3333
    # Explicit env vars also win for reasoning models.
    assert PipelineService._segment_max_tokens("Kimi-K2.6") == 1111
    assert PipelineService._final_max_tokens("Kimi-K3") == 3333


def test_reasoning_models_get_larger_default_budgets(monkeypatch):
    for name in (
        "LLM_NOTES_SEGMENT_MAX_TOKENS",
        "LLM_NOTES_REDUCE_MAX_TOKENS",
        "LLM_NOTES_FINAL_MAX_TOKENS",
        "LLM_REASONING_MODEL_PREFIXES",
    ):
        monkeypatch.delenv(name, raising=False)

    assert PipelineService._segment_max_tokens("Kimi-K2.6") == 16_000
    assert PipelineService._reduce_max_tokens("Kimi-K3") == 16_000
    assert PipelineService._final_max_tokens("kimi-k3") == 24_000
    assert PipelineService._segment_max_tokens("gpt-4o") == 4_000
    assert PipelineService._final_max_tokens("claude-opus-4-6") == 8_000


def test_length_retry_budget_is_capped(monkeypatch, pipeline):
    monkeypatch.setenv("LLM_NOTES_FALLBACK_MODEL", "")
    monkeypatch.setenv("LLM_NOTES_LENGTH_RETRY_TOKEN_MULTIPLIER", "3")
    monkeypatch.setenv("LLM_NOTES_LENGTH_RETRY_MAX_TOKENS", "20000")
    budgets: list[int] = []

    def always_limited(messages, **kwargs):
        budgets.append(kwargs["max_tokens"])
        raise _output_limit_error(stage=kwargs["stage"], max_tokens=kwargs["max_tokens"])

    pipeline.llm_service.call_llm = always_limited
    with pytest.raises(LLMOutputLimitError):
        pipeline._call_llm_with_length_retry(
            [{"role": "user", "content": "meeting"}],
            stage="segment_summary",
            max_tokens=16000,
            base_url="http://localhost:4000",
            api_key="k",
            model="Kimi-K2.6",
        )

    assert budgets == [16000, 20000]


def test_fallback_model_used_after_primary_exhausts_budget(monkeypatch, pipeline):
    monkeypatch.delenv("LLM_NOTES_FALLBACK_MODEL", raising=False)
    calls: list[dict] = []

    def fake_call(messages, **kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "Kimi-K2.6":
            raise _output_limit_error(stage=kwargs["stage"], max_tokens=kwargs["max_tokens"])
        return "fallback notes"

    pipeline.llm_service.call_llm = fake_call
    events: list[str] = []
    result = pipeline._call_llm_with_length_retry(
        [{"role": "user", "content": "meeting"}],
        stage="final_notes",
        max_tokens=1000,
        base_url="http://localhost:4000",
        api_key="k",
        model="Kimi-K2.6",
        events=events,
    )

    assert result == "fallback notes"
    assert [call["model"] for call in calls] == ["Kimi-K2.6", "Kimi-K2.6", "gpt-4o"]
    assert len(events) == 1 and "gpt-4o" in events[0]


def test_fallback_model_not_used_for_non_length_errors(monkeypatch, pipeline):
    monkeypatch.delenv("LLM_NOTES_FALLBACK_MODEL", raising=False)
    models: list[str] = []

    def fail(messages, **kwargs):
        models.append(kwargs["model"])
        raise RuntimeError("401 unauthorized")

    pipeline.llm_service.call_llm = fail
    with pytest.raises(RuntimeError, match="401"):
        pipeline._call_llm_with_length_retry(
            [{"role": "user", "content": "meeting"}],
            stage="final_notes",
            max_tokens=1000,
            base_url="http://localhost:4000",
            api_key="k",
            model="Kimi-K2.6",
        )

    assert models == ["Kimi-K2.6"]


def test_fallback_failure_reraises_original_length_error(monkeypatch, pipeline):
    monkeypatch.delenv("LLM_NOTES_FALLBACK_MODEL", raising=False)

    def fake_call(messages, **kwargs):
        if kwargs["model"] == "gpt-4o":
            raise RuntimeError("model not found")
        raise _output_limit_error(stage=kwargs["stage"], max_tokens=kwargs["max_tokens"])

    pipeline.llm_service.call_llm = fake_call
    with pytest.raises(LLMOutputLimitError):
        pipeline._call_llm_with_length_retry(
            [{"role": "user", "content": "meeting"}],
            stage="final_notes",
            max_tokens=1000,
            base_url="http://localhost:4000",
            api_key="k",
            model="Kimi-K2.6",
        )


def _segment_call_kwargs() -> dict:
    return {
        "base_url": "http://localhost:4000",
        "api_key": "k",
        "model": "Kimi-K2.6",
    }


def test_segment_is_split_when_primary_keeps_hitting_limit(monkeypatch, pipeline):
    monkeypatch.delenv("LLM_NOTES_FALLBACK_MODEL", raising=False)
    calls: list[dict] = []

    def fake_call(messages, **kwargs):
        calls.append(kwargs)
        if kwargs["stage"] == "segment_summary":
            raise _output_limit_error(stage=kwargs["stage"], max_tokens=kwargs["max_tokens"])
        return f"### Segment part ({kwargs['model']})"

    pipeline.llm_service.call_llm = fake_call
    events: list[str] = []
    segment_text = "First half of the meeting.\n\nSecond half of the meeting."
    summary = pipeline._summarize_segment_resilient(
        segment_index=2,
        total_segments=3,
        segment_text=segment_text,
        output_language="English",
        segment_max_tokens=4000,
        call_kwargs=_segment_call_kwargs(),
        events=events,
    )

    stages = [call["stage"] for call in calls]
    # Primary + one retry, never the fallback model for the whole segment.
    assert stages[:2] == ["segment_summary", "segment_summary"]
    assert all(call["model"] == "Kimi-K2.6" for call in calls[:2])
    assert stages[2:] == ["segment_summary_split", "segment_summary_split"]
    assert summary.count("### Segment part (Kimi-K2.6)") == 2
    assert any("smaller parts" in event for event in events)


def test_segment_falls_back_to_raw_text_when_everything_fails(monkeypatch, pipeline):
    monkeypatch.delenv("LLM_NOTES_FALLBACK_MODEL", raising=False)
    models: list[str] = []

    def always_limited(messages, **kwargs):
        models.append(kwargs["model"])
        raise _output_limit_error(stage=kwargs["stage"], max_tokens=kwargs["max_tokens"])

    pipeline.llm_service.call_llm = always_limited
    events: list[str] = []
    segment_text = "Alpha discussion.\n\nBeta decision."
    summary = pipeline._summarize_segment_resilient(
        segment_index=1,
        total_segments=1,
        segment_text=segment_text,
        output_language="English",
        segment_max_tokens=4000,
        call_kwargs=_segment_call_kwargs(),
        events=events,
    )

    assert "raw transcript – summary unavailable" in summary
    assert "Alpha discussion." in summary and "Beta decision." in summary
    assert "gpt-4o" in models  # fallback model was attempted for the split half
    assert any("raw transcript text" in event for event in events)


def test_split_segment_in_half_prefers_paragraph_boundary():
    text = "aaaa aaaa aaaa.\n\nbbbb bbbb bbbb."
    halves = PipelineService._split_segment_in_half(text)
    assert halves == ["aaaa aaaa aaaa.", "bbbb bbbb bbbb."]


def test_map_reduce_survives_failing_segment_and_reports_recovery(monkeypatch, pipeline, tmp_path):
    long_transcript = "\n\n".join(
        [
            "[Chunk 1]\nKickoff discussion and goals for the quarterly planning review.",
            "[Chunk 2]\nDecision: Use segmented transcript summarization for long meetings.",
        ]
    )
    transcript_path = tmp_path / "flaky_meeting.transcript.txt"
    transcript_path.write_text(long_transcript, encoding="utf-8")
    monkeypatch.setenv("LLM_NOTES_MAP_REDUCE_TRIGGER_CHARS", "100")
    monkeypatch.setenv("LLM_NOTES_MAP_REDUCE_SEGMENT_CHARS", "500")
    monkeypatch.setenv("LLM_NOTES_FALLBACK_MODEL", "")

    def fake_call_llm(messages, **kwargs):
        system_prompt = (kwargs.get("system_prompt") or "").lower()
        if "transcript segment analyst" in system_prompt:
            raise _output_limit_error(stage=kwargs["stage"], max_tokens=kwargs["max_tokens"])
        if "meeting transcript synthesizer" in system_prompt:
            return "## Condensed Transcript Digest\n### Coverage Notes\n- Raw segment used."
        return "## Executive Summary\nNotes despite segment failure."

    pipeline.llm_service.call_llm = fake_call_llm
    _, notes_text, _, _ = pipeline.run_pipeline(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        "Kimi-K2.6",
        interval_seconds=30,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(transcript_path),
    )

    assert "Notes despite segment failure." in notes_text
    assert "LLM output-limit recovery" in notes_text
    assert "raw transcript text" in notes_text


@pytest.mark.e2e
def test_pipeline_executes_visual_and_audio_branches_in_parallel(
    monkeypatch, pipeline, synthetic_video
):
    import threading

    branch_barrier = threading.Barrier(2, timeout=3.0)
    orig_extract = pipeline.video_service.extract_frames
    orig_transcribe = pipeline.video_service.transcribe_video_chunked

    def synced_extract(*args, **kwargs):
        res = orig_extract(*args, **kwargs)
        branch_barrier.wait()
        return res

    def synced_transcribe(*args, **kwargs):
        branch_barrier.wait()
        return orig_transcribe(*args, **kwargs)

    monkeypatch.setattr(pipeline.video_service, "extract_frames", synced_extract)
    monkeypatch.setattr(pipeline.video_service, "transcribe_video_chunked", synced_transcribe)

    thumbnails, notes_text, html_notes, html_path = _run(pipeline, synthetic_video)
    assert len(thumbnails) == 2
    assert "Executive Summary" in notes_text
    assert Path(html_path).exists()


@pytest.mark.e2e
def test_document_model_ocr_processes_frames_in_parallel_and_preserves_contextvar(
    monkeypatch, pipeline, synthetic_video
):
    import threading
    from meeting_notes.llm import LLMService

    monkeypatch.setenv("LLM_OCR_MODEL", "mistral-document-ai-2505")
    ocr_barrier = threading.Barrier(2, timeout=3.0)
    observed_efforts: list[str | None] = []
    lock = threading.Lock()
    notes_reply = pipeline.llm_service.reply

    def fake_call_llm(messages, **kwargs):
        prompt = (kwargs.get("system_prompt") or "").lower()
        if "ocr extraction assistant" in prompt:
            with lock:
                observed_efforts.append(LLMService.current_reasoning_effort())
            ocr_barrier.wait()
            return "Parallel OCR text"
        return notes_reply

    pipeline.llm_service.call_llm = fake_call_llm

    pipeline.run_pipeline(
        str(synthetic_video),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        reasoning_effort="low",
    )

    assert observed_efforts == ["low", "low"]
