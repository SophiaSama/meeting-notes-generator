"""Seam tests for Obsidian Vault Export (Tickets 01, 02 & 03)."""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from PIL import Image

from meeting_notes.llm import LLMOutputLimitError, LLMResponse, LLMService
from meeting_notes.notes import NotesService
from meeting_notes.pipeline import MeetingContext, PipelineService
from meeting_notes.video import VideoService
from tests.conftest import CANNED_NOTES, FakeLLMService, FakeVideoService

PROXY_ARGS = dict(
    base_url="http://localhost:4000",
    api_key="test-key",
    model="gpt-4o",
)


def _make_frame(index: int, time_str: str) -> dict:
    img = Image.new("RGB", (64, 48), color=(30 * index, 80, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "index": index,
        "time_str": time_str,
        "pil_image": img,
        "base64_jpeg": b64,
    }


def test_notes_service_default_build_markdown_notes_unchanged(tmp_path: Path):
    notes_service = NotesService()
    minutes_dir = tmp_path / "meeting_minutes"
    images_dir = minutes_dir / "images"
    minutes_dir.mkdir(parents=True)

    frames = [_make_frame(1, "0:02"), _make_frame(2, "0:04")]
    info = {"filename": "team sync #1.mp4", "duration_str": "0:06"}
    raw = "## Summary\nFirst\n@@SCREENSHOT_1@@\nSecond\n@@SCREENSHOT_2@@\nUnused @@SCREENSHOT_99@@"

    rendered = notes_service.build_markdown_notes(raw, frames, info, images_dir, minutes_dir)

    assert "@@SCREENSHOT_" not in rendered
    assert "![Screenshot at 0:02](images/team%20sync%20%231/screenshot_01.jpg)\n\n*Screenshot at 0:02*" in rendered
    assert "![Screenshot at 0:04](images/team%20sync%20%231/screenshot_02.jpg)\n\n*Screenshot at 0:04*" in rendered
    assert (images_dir / "team sync #1" / "screenshot_01.jpg").is_file()
    assert (images_dir / "team sync #1" / "screenshot_02.jpg").is_file()


def test_notes_service_parameterized_build_markdown_notes_custom_dir_and_prefix(tmp_path: Path):
    notes_service = NotesService()
    export_dir = tmp_path / "2026-11-22 Q3 Review"
    export_dir.mkdir(parents=True)
    images_dir = tmp_path / "unused_images"

    frames = [_make_frame(1, "0:02")]
    info = {"filename": "team_sync.mp4", "duration_str": "0:06"}
    raw = "## Summary\n@@SCREENSHOT_1@@\nExtra @@SCREENSHOT_2@@"

    rendered = notes_service.build_markdown_notes(
        raw,
        frames,
        info,
        images_dir,
        export_dir,
        attachment_dir="attachment",
        filename_prefix="Q3 Review #1",
    )

    assert "@@SCREENSHOT_" not in rendered
    assert "![Screenshot at 0:02](attachment/Q3%20Review%20%231_01.jpg)\n\n*Screenshot at 0:02*" in rendered
    assert (export_dir / "attachment" / "Q3 Review #1_01.jpg").is_file()
    assert not images_dir.exists()

    # Zero frames must not create an empty attachment/ or images/<stem> directory.
    empty_export_dir = tmp_path / "empty_export"
    empty_export_dir.mkdir(parents=True)
    rendered_empty = notes_service.build_markdown_notes(
        "## Summary\n@@SCREENSHOT_1@@",
        [],
        info,
        images_dir,
        empty_export_dir,
        attachment_dir="attachment",
        filename_prefix="Q3 Review #1",
    )
    assert "@@SCREENSHOT_" not in rendered_empty
    assert not (empty_export_dir / "attachment").exists()

    default_empty_images_dir = empty_export_dir / "images"
    rendered_default_empty = notes_service.build_markdown_notes(
        "## Summary\n@@SCREENSHOT_1@@",
        [],
        info,
        default_empty_images_dir,
        empty_export_dir,
    )
    assert "@@SCREENSHOT_" not in rendered_default_empty
    assert not (default_empty_images_dir / "team_sync").exists()


def test_notes_service_rewrite_screenshot_placeholders_from_frames_metadata():
    notes_service = NotesService()
    frames_meta = [
        {"index": 1, "time_str": "0:02", "ocr_text": "Slide 1", "ocr_keywords": ["slide"]},
        {"index": 10, "time_str": ""},
    ]
    raw = (
        "# Q4 Planning\n"
        "First @@SCREENSHOT_1@@\n"
        "Tenth @@SCREENSHOT_10@@\n"
        "Zero @@SCREENSHOT_0@@ Missing @@SCREENSHOT_99@@"
    )

    # Custom directory & prefix with special characters (spaces, #, parentheses)
    rewritten = notes_service.rewrite_screenshot_placeholders(
        raw,
        frames_meta,
        attachment_dir="media/Q4 Shots",
        filename_prefix="Q4 Review #1 (Draft)",
    )
    assert "@@SCREENSHOT_" not in rewritten
    assert (
        "![Screenshot at 0:02](media/Q4%20Shots/Q4%20Review%20%231%20%28Draft%29_01.jpg)"
        "\n\n*Screenshot at 0:02*"
    ) in rewritten
    # Missing time_str falls back cleanly to "Screenshot {idx}", and _1 does not collide with _10
    assert (
        "![Screenshot 10](media/Q4%20Shots/Q4%20Review%20%231%20%28Draft%29_10.jpg)"
        "\n\n*Screenshot 10*"
    ) in rewritten

    # Blank/whitespace prefix falls back to "screenshot" and empty/dot attachment_dir emits bare filename
    bare = notes_service.rewrite_screenshot_placeholders(
        "@@SCREENSHOT_1@@",
        [{"index": 1, "time_str": "0:05"}],
        attachment_dir=".",
        filename_prefix="   ",
    )
    assert bare == "![Screenshot at 0:05](screenshot_01.jpg)\n\n*Screenshot at 0:05*"

    bare_empty_dir = notes_service.rewrite_screenshot_placeholders(
        "@@SCREENSHOT_1@@",
        [{"index": 1, "time_str": "0:05"}],
        attachment_dir="",
        filename_prefix="",
    )
    assert bare_empty_dir == "![Screenshot at 0:05](screenshot_01.jpg)\n\n*Screenshot at 0:05*"

    # Zero frames strips all placeholders cleanly
    stripped = notes_service.rewrite_screenshot_placeholders("Keep text @@SCREENSHOT_1@@ done", [])
    assert stripped == "Keep text  done"


def test_notes_service_build_html_notes_with_theme_unchanged():
    notes_service = NotesService()
    frames = [_make_frame(1, "0:02")]
    info = {"filename": "team_sync.mp4", "duration_str": "0:06"}
    raw = "## Summary\nHello\n@@SCREENSHOT_1@@\n@@SCREENSHOT_99@@"

    html = notes_service.build_html_notes(raw, frames, info, "gpt-4o", theme="ink-wash")
    assert html.startswith("<!DOCTYPE html>")
    assert 'data-art-theme="ink-wash"' in html
    assert '<figure class="screenshot"><img src="data:image/jpeg;base64,' in html
    assert "@@SCREENSHOT_" not in html


@pytest.mark.e2e
def test_run_pipeline_recording_retains_raw_notes_and_screenshots_dir(pipeline, synthetic_video):
    thumbnails, notes_text, html_notes, html_path, ctx = pipeline.run_pipeline(
        str(synthetic_video),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        return_context=True,
    )

    assert len(thumbnails) == 2
    assert "Artifacts" in notes_text
    assert Path(html_path).exists()
    assert isinstance(ctx, MeetingContext)

    # raw_notes retains clean Provider Markdown with @@SCREENSHOT_N@@ placeholders
    assert ctx["raw_notes"] == CANNED_NOTES.strip()
    assert ctx.raw_notes == CANNED_NOTES.strip()
    assert "@@SCREENSHOT_1@@" in ctx["raw_notes"]
    assert "@@SCREENSHOT_2@@" in ctx["raw_notes"]
    assert "Artifacts" not in ctx["raw_notes"]
    assert "# Meeting Minutes" not in ctx["raw_notes"]
    assert "images/" not in ctx["raw_notes"]

    # markdown_notes still holds rewritten internal image links for Meeting Chat
    assert "![Screenshot at" in ctx["markdown_notes"]
    assert "@@SCREENSHOT_" not in ctx["markdown_notes"]

    # screenshots_dir is a JSON-serializable str pointing to the saved screenshot files
    assert isinstance(ctx["screenshots_dir"], str)
    assert ctx.screenshots_dir == ctx["screenshots_dir"]
    shot_dir = Path(ctx["screenshots_dir"])
    assert shot_dir.is_dir()
    assert (shot_dir / "screenshot_01.jpg").is_file()
    assert (shot_dir / "screenshot_02.jpg").is_file()

    roundtrip = json.loads(json.dumps(ctx))
    assert roundtrip["raw_notes"] == ctx["raw_notes"]
    assert roundtrip["screenshots_dir"] == ctx["screenshots_dir"]


@pytest.mark.e2e
def test_run_pipeline_transcript_only_sets_screenshots_dir_none_and_retains_raw_notes(
    pipeline, tmp_path: Path
):
    transcript_reply = "# Sprint Retro\n\n## Summary\nReviewed sprint metrics.\n"
    pipeline.llm_service.reply = transcript_reply
    tx_file = tmp_path / "sprint_retro.txt"
    tx_file.write_text("Alice: Sprint metrics look good.\n", encoding="utf-8")

    thumbnails, notes_text, _, _, ctx = pipeline.run_pipeline(
        None,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        transcript_file=str(tx_file),
        return_context=True,
    )

    assert thumbnails == []
    assert "Artifacts" in notes_text
    assert ctx["screenshots_dir"] is None
    assert ctx["frames_metadata"] == []
    assert ctx["raw_notes"] == transcript_reply.strip()
    assert "Artifacts" not in ctx["raw_notes"]
    assert "# Meeting Minutes" not in ctx["raw_notes"]

    roundtrip = json.loads(json.dumps(ctx))
    assert roundtrip["screenshots_dir"] is None
    assert roundtrip["raw_notes"] == transcript_reply.strip()

    # No empty images/<stem> folder created on transcript-only runs
    assert not (Path.cwd() / "artifacts" / "meeting_minutes" / "images" / "sprint_retro").exists()


@pytest.mark.e2e
def test_run_pipeline_zero_screenshots_recording_sets_screenshots_dir_none(
    monkeypatch, pipeline, synthetic_video
):
    monkeypatch.setattr(pipeline.video_service, "extract_frames", lambda *args, **kwargs: [])

    thumbnails, _, _, _, ctx = pipeline.run_pipeline(
        str(synthetic_video),
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        return_context=True,
    )

    assert thumbnails == []
    assert ctx["frames_metadata"] == []
    assert ctx["screenshots_dir"] is None
    assert ctx["raw_notes"] == CANNED_NOTES.strip()
    assert "@@SCREENSHOT_" not in ctx["markdown_notes"]
    assert not (Path.cwd() / "artifacts" / "meeting_minutes" / "images").exists()


def test_build_meeting_context_backward_compatibility_and_normalization(tmp_path: Path):
    legacy_ctx = PipelineService.build_meeting_context(
        info={"filename": "sync.mp4"},
        markdown_notes="## Notes\nDone",
        transcript_evidence="[00:01] Hi",
        transcript_source_label="transcript",
        frames=[],
        output_language="English",
        notes_style="Executive Summary",
    )
    assert legacy_ctx["raw_notes"] == ""
    assert legacy_ctx["screenshots_dir"] is None
    assert json.loads(json.dumps(legacy_ctx))["screenshots_dir"] is None

    shots_path = tmp_path / "shots"
    path_ctx = PipelineService.build_meeting_context(
        info={"filename": "sync.mp4"},
        markdown_notes="## Notes\nDone",
        transcript_evidence="[00:01] Hi",
        transcript_source_label="transcript",
        frames=[],
        output_language="English",
        notes_style="Executive Summary",
        raw_notes="  # Sync Title\n\nBody @@SCREENSHOT_1@@  ",
        screenshots_dir=shots_path,
    )
    assert path_ctx["raw_notes"] == "# Sync Title\n\nBody @@SCREENSHOT_1@@"
    assert path_ctx["screenshots_dir"] == str(shots_path)
    assert isinstance(path_ctx["screenshots_dir"], str)
    assert json.loads(json.dumps(path_ctx))["screenshots_dir"] == str(shots_path)

    blank_ctx = PipelineService.build_meeting_context(
        info={"filename": "sync.mp4"},
        markdown_notes="## Notes",
        transcript_evidence="",
        transcript_source_label="transcript",
        frames=[],
        output_language="English",
        notes_style="Executive Summary",
        screenshots_dir="   ",
    )
    assert blank_ctx["screenshots_dir"] is None


def _make_obsidian_pipeline(reply: str = CANNED_NOTES) -> PipelineService:
    llm = FakeLLMService(reply=reply)
    video = FakeVideoService(llm)
    return PipelineService(llm, video, NotesService())


def _write_test_jpegs(directory: Path, count: int = 2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        img = Image.new("RGB", (48, 36), color=(20 * idx, 100, 180))
        img.save(directory / f"screenshot_{idx:02d}.jpg", format="JPEG")
    return directory


def test_prepare_obsidian_export_validates_vault_before_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pipeline = _make_obsidian_pipeline()
    regular_file = tmp_path / "vault.txt"
    regular_file.write_text("not a dir", encoding="utf-8")

    invalid_vaults = [
        None,
        "",
        "   ",
        f"{tmp_path}\x00bad",
        tmp_path / "does_not_exist",
        regular_file,
    ]
    for bad_vault in invalid_vaults:
        with pytest.raises((ValueError, OSError)):
            pipeline.prepare_obsidian_export(
                bad_vault,  # type: ignore[arg-type]
                raw_notes="## Summary\nNo H1 heading here.",
                source_stem="weekly",
                **PROXY_ARGS,
            )
    assert pipeline.llm_service.calls == []

    ro_vault = tmp_path / "ro_vault"
    ro_vault.mkdir()
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        ro_vault.chmod(0o555)
        try:
            with pytest.raises(PermissionError):
                pipeline.prepare_obsidian_export(
                    ro_vault,
                    raw_notes="## Summary\nNo H1.",
                    source_stem="weekly",
                    **PROXY_ARGS,
                )
        finally:
            ro_vault.chmod(0o755)

    writable_vault = tmp_path / "writable_vault"
    writable_vault.mkdir()
    monkeypatch.setattr(os, "access", lambda *_args, **_kwargs: False)
    with pytest.raises(PermissionError):
        pipeline.prepare_obsidian_export(
            writable_vault,
            raw_notes="## Summary\nNo H1.",
            source_stem="weekly",
            **PROXY_ARGS,
        )
    assert pipeline.llm_service.calls == []

    monkeypatch.undo()

    def _deny_temp(*_args, **_kwargs):
        raise PermissionError("ACL denied")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _deny_temp)
    with pytest.raises(PermissionError):
        pipeline.prepare_obsidian_export(
            writable_vault,
            raw_notes="## Summary\nNo H1.",
            source_stem="weekly",
            **PROXY_ARGS,
        )
    assert pipeline.llm_service.calls == []

    monkeypatch.undo()
    real_vault = tmp_path / "real_vault"
    real_vault.mkdir()
    link_vault = tmp_path / "link_vault"
    link_vault.symlink_to(real_vault)
    prepared = pipeline.prepare_obsidian_export(
        link_vault,
        raw_notes="# Symlinked Vault\nBody.",
        source_stem="sym",
        export_date="2026-11-22",
    )
    assert Path(prepared["target_dir"]).parent == link_vault
    assert list(real_vault.iterdir()) == []


def test_prepare_obsidian_export_title_from_h1_ignores_h2_and_fenced_code_skips_provider(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()
    notes = (
        "```python\n"
        "# python comment inside backtick fence\n"
        "```\n\n"
        "~~~\n"
        "# comment inside tilde fence\n"
        "~~~\n\n"
        "## Overview\n"
        "Intro text.\n\n"
        "# [Q3](https://example.com) **Architecture** & API Review: v2.1 @@SCREENSHOT_1@@\n\n"
        "# Second H1 Ignored\n"
    )

    prepared = pipeline.prepare_obsidian_export(
        vault,
        raw_notes=notes,
        source_stem="recording_01",
        export_date="2026-11-22",
        **PROXY_ARGS,
    )

    assert pipeline.llm_service.calls == []
    assert prepared["folder_name"] == "2026-11-22 Q3 Architecture & API Review v2.1"
    assert prepared["notes_filename"] == "2026-11-22 Q3 Architecture & API Review v2.1.md"
    assert prepared["faq_filename"] == "2026-11-22 Q3 Architecture & API Review v2.1 - FAQ.md"
    assert prepared["exists"] is False
    assert prepared["faq_markdown"] is None
    assert prepared["default_attachment_prefix"] == "screenshot"
    assert json.loads(json.dumps(prepared))["folder_name"] == prepared["folder_name"]
    assert list(vault.iterdir()) == []


def test_prepare_obsidian_export_title_provider_fallback_respects_reasoning_effort_and_length_retry(
    tmp_path: Path,
):
    class _RetryTitleLLM(FakeLLMService):
        def __init__(self):
            super().__init__()
            self.observed_efforts: list[str | None] = []

        def call_llm(self, messages, **kwargs):
            self.observed_efforts.append(self.current_reasoning_effort())
            self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
            if kwargs.get("attempt") == 1:
                requested = int(kwargs.get("max_tokens") or 256)
                raise LLMOutputLimitError(
                    model=str(kwargs.get("model") or "o3-mini"),
                    requested_max_tokens=requested,
                    response=LLMResponse(
                        content="",
                        finish_reason="length",
                        output_tokens=requested,
                    ),
                    stage=str(kwargs.get("stage") or ""),
                )
            return '"Title: Platform Migration Sync."'

    llm = _RetryTitleLLM()
    pipeline = PipelineService(llm, FakeVideoService(llm), NotesService())
    vault = tmp_path / "vault"
    vault.mkdir()

    prepared = pipeline.prepare_obsidian_export(
        vault,
        raw_notes=CANNED_NOTES,
        source_stem="fallback_stem",
        base_url="http://localhost:4000",
        api_key="test-key",
        model="o3-mini",
        reasoning_effort="high",
        export_date="2026-11-22",
    )

    assert len(llm.calls) == 2
    assert llm.observed_efforts == ["high", "high"]
    assert llm.calls[0]["kwargs"]["stage"] == "obsidian_title"
    assert llm.calls[1]["kwargs"]["stage"] == "obsidian_title"
    assert llm.calls[1]["kwargs"]["max_tokens"] > llm.calls[0]["kwargs"]["max_tokens"]
    assert prepared["folder_name"] == "2026-11-22 Platform Migration Sync"


def test_prepare_obsidian_export_title_falls_back_to_source_stem_on_provider_failure_or_blank(
    tmp_path: Path,
):
    vault = tmp_path / "vault"
    vault.mkdir()

    p_fail = _make_obsidian_pipeline()
    p_fail.llm_service.call_llm = lambda *_a, **_kw: (_ for _ in ()).throw(
        RuntimeError("timeout")
    )
    prep_a = p_fail.prepare_obsidian_export(
        vault,
        raw_notes="## Summary\nNo H1.",
        source_stem="weekly_sync_2026",
        export_date="2026-11-22",
        **PROXY_ARGS,
    )
    assert prep_a["folder_name"] == "2026-11-22 weekly_sync_2026"

    p_blank = _make_obsidian_pipeline()
    p_blank.llm_service.reply = "   \n  "
    prep_b = p_blank.prepare_obsidian_export(
        vault,
        meeting_context={
            "info": {"filename": "team:standup..v1.mp4"},
            "raw_notes": "## Summary\nOk",
        },
        export_date="2026-11-22",
        **PROXY_ARGS,
    )
    assert prep_b["folder_name"] == "2026-11-22 team standup v1"

    p_dots = _make_obsidian_pipeline()
    p_dots.llm_service.reply = "..."
    prep_c = p_dots.prepare_obsidian_export(
        vault,
        raw_notes="# ???\nBody",
        source_stem="..",
        export_date="2026-11-22",
        **PROXY_ARGS,
    )
    assert prep_c["folder_name"] == "2026-11-22 Meeting Notes"


def test_prepare_and_export_sanitization_long_titles_windows_reserved_and_attachment_prefix(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()
    shots = _write_test_jpegs(tmp_path / "shots", count=1)

    prep_traversal = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# ../../etc\\passwd: *Draft* <v1.2>?\nBody",
        source_stem="s",
        export_date="2026-11-22",
    )
    assert prep_traversal["folder_name"] == "2026-11-22 etc passwd Draft v1.2"
    assert Path(prep_traversal["target_dir"]).parent == vault

    prep_reserved = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# CON\nBody",
        source_stem="s",
        export_date="2026-11-22",
    )
    assert prep_reserved["folder_name"] == "2026-11-22 CON_"

    prep_long_en = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# " + ("VeryLongSegment " * 12),
        source_stem="s",
        export_date="2026-11-22",
    )
    assert len(prep_long_en["folder_name"]) <= 140
    assert len(prep_long_en["notes_filename"]) <= 150

    prep_long_cjk = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# " + ("架构评审会议" * 20),
        source_stem="s",
        export_date="2026-11-22",
    )
    assert len(prep_long_cjk["title"].encode("utf-8")) <= 180
    assert len(prep_long_cjk["folder_name"]) <= 140

    # Prefix variants: blank/punctuation default to "screenshot"; safe #, (, ), spaces preserved.
    for blank_pref in ("", "   ", "../..///:**?"):
        assert pipeline._sanitize_attachment_prefix(blank_pref) == "screenshot"

    assert pipeline._sanitize_attachment_prefix("../clip:01*") == "clip 01"

    prep_prefix = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Prefix Check\n@@SCREENSHOT_1@@",
        source_stem="pref",
        screenshots_dir=shots,
        frames_metadata=[{"index": 1, "time_str": "0:02"}],
        export_date="2026-11-22",
    )
    res = pipeline.export_to_obsidian(
        prep_prefix, attachment_prefix="Q3 Review #1 (Draft)"
    )
    assert (
        Path(res["target_dir"]) / "attachment" / "Q3 Review #1 (Draft)_01.jpg"
    ).is_file()
    md_text = Path(res["notes_path"]).read_text(encoding="utf-8")
    assert "attachment/Q3%20Review%20%231%20%28Draft%29_01.jpg" in md_text

    with pytest.raises(ValueError):
        pipeline.export_to_obsidian(
            vault_path=vault,
            folder_name="../escape",
            raw_notes="# Escape\nBody",
        )


def test_prepare_and_export_collision_guard_overwrite_false_leaves_disk_identical(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()
    shots = _write_test_jpegs(tmp_path / "shots", count=1)

    prep = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Collision Test\nNew content @@SCREENSHOT_1@@",
        source_stem="col",
        screenshots_dir=shots,
        frames_metadata=[{"index": 1, "time_str": "0:02"}],
        export_date="2026-11-22",
    )
    target = Path(prep["target_dir"])
    (target / "attachment").mkdir(parents=True)
    notes_file = target / prep["notes_filename"]
    faq_file = target / prep["faq_filename"]
    old_shot = target / "attachment" / "old_01.jpg"
    personal = target / "personal.md"
    notes_file.write_text("EXISTING NOTES", encoding="utf-8")
    faq_file.write_text("EXISTING FAQ", encoding="utf-8")
    old_shot.write_bytes(b"OLD_SHOT_BYTES")
    personal.write_text("KEEP ME", encoding="utf-8")

    prep_existing = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Collision Test\nNew content",
        source_stem="col",
        export_date="2026-11-22",
    )
    assert prep_existing["exists"] is True

    conflict_res = pipeline.export_to_obsidian(
        prep, attachment_prefix="new", faq_markdown="NEW FAQ", overwrite=False
    )
    assert conflict_res["status"] == "conflict"
    assert conflict_res["written"] is False
    assert conflict_res["exported"] is False
    assert conflict_res["exists"] is True
    assert conflict_res["overwritten"] is False

    conflict_default = pipeline.export_to_obsidian(prep)
    assert conflict_default["status"] == "conflict"
    assert conflict_default["exported"] is False

    conflict_direct = pipeline.export_to_obsidian(
        vault_path=vault,
        folder_name=prep["folder_name"],
        raw_notes="# Collision Test\nDirect",
        overwrite=False,
    )
    assert conflict_direct["status"] == "conflict"

    assert notes_file.read_text(encoding="utf-8") == "EXISTING NOTES"
    assert faq_file.read_text(encoding="utf-8") == "EXISTING FAQ"
    assert old_shot.read_bytes() == b"OLD_SHOT_BYTES"
    assert personal.read_text(encoding="utf-8") == "KEEP ME"


@pytest.mark.e2e
def test_export_to_obsidian_nondestructive_copy_link_rewriting_and_zero_screenshots(
    pipeline: PipelineService, synthetic_video: Path, tmp_path: Path
):
    _, _, _, _, ctx = pipeline.run_pipeline(
        str(synthetic_video),
        **PROXY_ARGS,
        interval_seconds=2,
        max_frames=8,
        notes_style="Executive Summary",
        output_language="English",
        return_context=True,
    )
    shot_dir = Path(ctx["screenshots_dir"])
    shot_1_bytes = (shot_dir / "screenshot_01.jpg").read_bytes()
    shot_2_bytes = (shot_dir / "screenshot_02.jpg").read_bytes()

    vault = tmp_path / "vault"
    vault.mkdir()
    prepared = pipeline.prepare_obsidian_export(
        vault, meeting_context=ctx, export_date="2026-11-22"
    )
    assert prepared["screenshot_count"] == 2
    res = pipeline.export_to_obsidian(prepared, attachment_prefix="Q3 Review #1")

    assert res["status"] == "exported"
    assert res["written"] is True
    assert res["exported"] is True
    assert res["screenshots_copied"] == 2
    tdir = Path(res["target_dir"])
    assert (tdir / "attachment" / "Q3 Review #1_01.jpg").read_bytes() == shot_1_bytes
    assert (tdir / "attachment" / "Q3 Review #1_02.jpg").read_bytes() == shot_2_bytes
    exported_md = Path(res["notes_path"]).read_text(encoding="utf-8")
    assert "attachment/Q3%20Review%20%231_01.jpg" in exported_md
    assert "attachment/Q3%20Review%20%231_02.jpg" in exported_md
    assert "@@SCREENSHOT_" not in exported_md
    # Internal artifacts remain completely intact.
    assert (shot_dir / "screenshot_01.jpg").read_bytes() == shot_1_bytes
    assert (shot_dir / "screenshot_02.jpg").read_bytes() == shot_2_bytes

    # Rewriting internal images/<stem>/ links without frames_metadata & without duplicating captions.
    direct_res = pipeline.export_to_obsidian(
        vault_path=vault,
        notes_markdown=(
            "# Direct Export\n\n"
            "![Screenshot at 0:02](images/team%20sync%20%231/screenshot_01.jpg)\n\n"
            "*Screenshot at 0:02*\n"
        ),
        screenshots_dir=shot_dir,
        attachment_prefix="direct shot",
        export_date="2026-11-22",
    )
    direct_md = Path(direct_res["notes_path"]).read_text(encoding="utf-8")
    assert "![Screenshot at 0:02](attachment/direct%20shot_01.jpg)" in direct_md
    assert direct_md.count("*Screenshot at 0:02*") == 1

    # Zero-screenshot cases: no attachment/ directory created and placeholders stripped.
    empty_shots = tmp_path / "empty_shots"
    empty_shots.mkdir()
    for label, sdir in (
        ("none", None),
        ("missing", tmp_path / "missing_shots"),
        ("empty", empty_shots),
    ):
        zero_prep = pipeline.prepare_obsidian_export(
            vault,
            raw_notes=f"# Zero {label}\nBody @@SCREENSHOT_1@@",
            source_stem=f"zero_{label}",
            screenshots_dir=sdir,
            export_date="2026-11-22",
        )
        assert zero_prep["screenshot_count"] == 0
        zero_res = pipeline.export_to_obsidian(zero_prep, attachment_prefix="ignored")
        assert zero_res["attachment_dir"] is None
        assert not (Path(zero_res["target_dir"]) / "attachment").exists()
        assert "@@SCREENSHOT_" not in Path(zero_res["notes_path"]).read_text(
            encoding="utf-8"
        )


def test_export_to_obsidian_selective_overwrite_and_faq_sentinel(tmp_path: Path):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()
    shots = _write_test_jpegs(tmp_path / "shots", count=2)

    prep = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Selective Overwrite\nV1 @@SCREENSHOT_1@@ @@SCREENSHOT_2@@",
        source_stem="sel",
        screenshots_dir=shots,
        frames_metadata=[
            {"index": 1, "time_str": "0:02"},
            {"index": 2, "time_str": "0:04"},
        ],
        export_date="2026-11-22",
    )
    prep_with_faq = {**prep, "faq_markdown": "## Initial FAQ\n- Q: Status? A: Green."}
    prep_snapshot = copy.deepcopy(prep_with_faq)

    # Explicit faq_markdown=None suppresses FAQ even when prep carries faq_markdown.
    first = pipeline.export_to_obsidian(
        prep_with_faq, attachment_prefix="alpha", faq_markdown=None
    )
    tdir = Path(first["target_dir"])
    assert not (tdir / prep["faq_filename"]).exists()
    assert prep_with_faq == prep_snapshot

    # Add user files (including read-only and subdirectory) and stale attachment file.
    personal_md = tdir / "personal.md"
    personal_md.write_text("User notes", encoding="utf-8")
    same_prefix_root = tdir / "alpha_01.jpg"
    same_prefix_root.write_bytes(b"USER_IMAGE")
    sub_dir = tdir / "diagrams"
    sub_dir.mkdir()
    (sub_dir / "whiteboard.png").write_bytes(b"PNG")
    (tdir / "attachment" / "stale.txt").write_text("stale", encoding="utf-8")
    ro_user = tdir / "locked_user.md"
    ro_user.write_text("locked", encoding="utf-8")
    ro_user.chmod(0o444)

    try:
        second = pipeline.export_to_obsidian(
            prep_with_faq,
            attachment_prefix="beta",
            overwrite=True,
        )
        assert second["overwritten"] is True
        assert not (tdir / "attachment" / "alpha_01.jpg").exists()
        assert not (tdir / "attachment" / "stale.txt").exists()
        assert (tdir / "attachment" / "beta_01.jpg").is_file()
        assert (tdir / "attachment" / "beta_02.jpg").is_file()
        assert "Initial FAQ" in (tdir / prep["faq_filename"]).read_text(encoding="utf-8")
        assert personal_md.read_text(encoding="utf-8") == "User notes"
        assert same_prefix_root.read_bytes() == b"USER_IMAGE"
        assert (sub_dir / "whiteboard.png").read_bytes() == b"PNG"
        assert ro_user.read_text(encoding="utf-8") == "locked"

        # Overwriting with 0 screenshots and blank faq_markdown cleans attachment/ & stale FAQ.
        third = pipeline.export_to_obsidian(
            prep_with_faq,
            raw_notes="# Selective Overwrite\nTranscript only V3",
            screenshots_dir=None,
            faq_markdown="   ",
            overwrite=True,
        )
        assert third["attachment_dir"] is None
        assert third["faq_path"] is None
        assert not (tdir / "attachment").exists()
        assert not (tdir / prep["faq_filename"]).exists()
        assert "Transcript only V3" in (tdir / prep["notes_filename"]).read_text(
            encoding="utf-8"
        )
        assert personal_md.read_text(encoding="utf-8") == "User notes"
        assert (sub_dir / "whiteboard.png").read_bytes() == b"PNG"
        assert ro_user.read_text(encoding="utf-8") == "locked"
        assert prep_with_faq == prep_snapshot
    finally:
        ro_user.chmod(0o644)


# ---------------------------------------------------------------------------
# Ticket 03: Meeting Chat Filtering, Provider FAQ Distillation, & M2 Hardening
# ---------------------------------------------------------------------------


def test_m2_hardening_path_sanitization_headings_and_title_cleaning():
    assert PipelineService._sanitize_path_component("CON.v1") == "CON_.v1"
    assert PipelineService._sanitize_path_component("CON .v1") == "CON_.v1"
    assert PipelineService._sanitize_path_component("con.txt") == "con_.txt"
    assert PipelineService._clean_llm_title_response("Title: # Foo") == "Foo"
    assert PipelineService._clean_llm_title_response('Meeting Title: "## Bar."') == "Bar"
    assert (
        PipelineService._extract_first_h1_heading("# ###\n# Actual Title")
        == "Actual Title"
    )
    assert PipelineService._extract_first_h1_heading("# F#") == "F#"
    assert (
        PipelineService._extract_first_h1_heading("# Q3 Review #1 ###")
        == "Q3 Review #1"
    )


def test_prepare_obsidian_export_normalizes_all_chat_formats_and_filters_errors_and_orphans(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline(reply="## Frequently Asked Questions\n\n### Who owns rollout?\nAlice.")
    vault = tmp_path / "vault"
    vault.mkdir()

    mixed_history = [
        (
            "Orphaned Q1",
            "Please generate meeting notes first before asking questions about the meeting.",
        ),
        {"role": "user", "content": "Orphaned Q2"},
        {
            "role": "assistant",
            "content": (
                "Unable to answer the question right now. Please check your LLM settings and try again.\n\n"
                "Error: RuntimeError: timeout"
            ),
        },
        {"role": "user", "content": "Orphaned Q3"},
        {"role": "assistant", "content": "LLM base URL is required."},
        {"role": "user", "content": "Orphaned Q4"},
        {"role": "assistant", "content": "Model name is required."},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "Please enter a question before submitting."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Generate Meeting Notes first, then ask questions about the meeting here.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who owns rollout?"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Alice owns rollout and noted LLM base URL is required in config.",
                }
            ],
        },
        SimpleNamespace(role="user", content="When is freeze?"),
        SimpleNamespace(role="assistant", content="Friday at 17:00 UTC."),
        {"role": "user", "content": "Trailing unanswered"},
    ]

    prepared = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Release Readiness\nBody text.",
        source_stem="rel",
        chat_history=mixed_history,
        **PROXY_ARGS,
    )

    assert prepared["faq_markdown"] == "## Frequently Asked Questions\n\n### Who owns rollout?\nAlice."
    assert len(pipeline.llm_service.calls) == 1
    call = pipeline.llm_service.calls[0]
    assert call["kwargs"]["stage"] == "obsidian_faq"
    assert "title" not in call["kwargs"]["stage"].lower()
    assert "title" not in (call["kwargs"].get("system_prompt") or "").lower()
    prompt_blob = json.dumps(call["messages"])
    assert "Who owns rollout?" in prompt_blob
    assert "Alice owns rollout and noted LLM base URL is required in config." in prompt_blob
    assert "When is freeze?" in prompt_blob
    assert "Friday at 17:00 UTC." in prompt_blob
    for dropped in (
        "Orphaned Q1",
        "Orphaned Q2",
        "Orphaned Q3",
        "Orphaned Q4",
        "Trailing unanswered",
        "Please generate meeting notes first",
        "Unable to answer the question right now",
        "Please enter a question",
        "Generate Meeting Notes first",
    ):
        assert dropped not in prompt_blob


@pytest.mark.parametrize(
    "chat_history",
    [
        None,
        [],
        "",
        "   ",
        "[]",
        [("", ""), ("   ", "   ")],
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": ""}],
        [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "Please generate meeting notes first before asking questions about the meeting.",
            },
        ],
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "LLM base URL is required."},
        ],
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Model name is required."},
        ],
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Error: ConnectionError: refused"},
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ],
            },
            {"role": "assistant", "content": "Done"},
        ],
        [{"role": "user", "content": "Unanswered only"}],
    ],
)
def test_prepare_obsidian_export_skips_provider_when_chat_empty_or_errors_only(
    tmp_path: Path, chat_history
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()

    # Empty/placeholder chat succeeds with 0 Provider calls even without LLM credentials.
    prepared = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Empty Chat Export\nBody.",
        source_stem="empty_chat",
        chat_history=chat_history,
        base_url="",
        api_key="",
        model="",
    )
    assert prepared["faq_markdown"] is None
    assert pipeline.llm_service.calls == []

    exported = pipeline.export_to_obsidian(prepared)
    assert exported["faq_path"] is None
    assert not (Path(exported["target_dir"]) / prepared["faq_filename"]).exists()


@pytest.mark.parametrize("bad_chat", ["not-json", "{", '{"role": "user"}', "123"])
def test_prepare_obsidian_export_rejects_invalid_json_string_chat_history(
    tmp_path: Path, bad_chat: str
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()

    with pytest.raises(ValueError, match="chat_history must be a JSON list"):
        pipeline.prepare_obsidian_export(
            vault,
            raw_notes="# Bad Chat\nBody.",
            source_stem="bad_chat",
            chat_history=bad_chat,
            **PROXY_ARGS,
        )
    assert pipeline.llm_service.calls == []


def test_prepare_obsidian_export_accepts_json_string_and_whitespace_reply_returns_none(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline(reply="## FAQ\n\n### Status?\nGreen.")
    vault = tmp_path / "vault"
    vault.mkdir()

    prepared = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# JSON Chat\nBody.",
        source_stem="json_chat",
        chat_history=json.dumps(
            [{"role": "user", "content": "Status?"}, {"role": "assistant", "content": "Green."}]
        ),
        **PROXY_ARGS,
    )
    assert prepared["faq_markdown"] == "## FAQ\n\n### Status?\nGreen."
    assert len(pipeline.llm_service.calls) == 1

    # Whitespace-only Provider reply normalizes to None and writes no FAQ file.
    ws_pipeline = _make_obsidian_pipeline(reply="   \n\t  ")
    ws_prep = ws_pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Whitespace FAQ\nBody.",
        source_stem="ws_faq",
        chat_history=[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}],
        **PROXY_ARGS,
    )
    assert ws_prep["faq_markdown"] is None
    ws_export = ws_pipeline.export_to_obsidian(ws_prep)
    assert ws_export["faq_path"] is None
    assert not (Path(ws_export["target_dir"]) / ws_prep["faq_filename"]).exists()


def test_prepare_obsidian_export_preserves_outer_reasoning_effort_and_budgets(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()
    seen: list[tuple[str, str | None, int]] = []

    def tracking_call(messages, **kwargs):
        seen.append(
            (
                str(kwargs.get("stage")),
                pipeline.llm_service.current_reasoning_effort(),
                int(kwargs["max_tokens"]),
            )
        )
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        if kwargs.get("stage") == "obsidian_title":
            return "Outer Reasoning Title"
        return "## FAQ\n\n### Why?\nBecause."

    pipeline.llm_service.call_llm = tracking_call

    # Outer ContextVar "high" is preserved for both title fallback and FAQ when reasoning_effort=None.
    with pipeline.llm_service.reasoning_effort("high"):
        prep_inherited = pipeline.prepare_obsidian_export(
            vault,
            raw_notes="## Summary\nNo H1 heading here.",
            source_stem="ctx_inherit",
            chat_history=[{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."}],
            reasoning_effort=None,
            **PROXY_ARGS,
        )
    assert prep_inherited["title"] == "Outer Reasoning Title"
    assert seen == [
        ("obsidian_title", "high", PipelineService.DEFAULT_REASONING_OBSIDIAN_TITLE_MAX_TOKENS),
        ("obsidian_faq", "high", PipelineService.DEFAULT_REASONING_OBSIDIAN_FAQ_MAX_TOKENS),
    ]

    # Explicit reasoning_effort="none" overrides outer "high" and uses non-reasoning token budget.
    seen.clear()
    with pipeline.llm_service.reasoning_effort("high"):
        pipeline.prepare_obsidian_export(
            vault,
            raw_notes="# Explicit None\nBody.",
            source_stem="explicit_none",
            chat_history=[{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."}],
            reasoning_effort="none",
            **PROXY_ARGS,
        )
    assert seen == [
        ("obsidian_faq", "none", PipelineService.DEFAULT_OBSIDIAN_FAQ_MAX_TOKENS),
    ]


def test_prepare_obsidian_export_faq_length_retry_and_language_and_budget_trimming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()
    attempts: list[int] = []

    def retry_call(messages, **kwargs):
        attempts.append(int(kwargs["max_tokens"]))
        pipeline.llm_service.calls.append({"messages": messages, "kwargs": kwargs})
        if len(attempts) == 1:
            raise LLMOutputLimitError(
                model=kwargs["model"],
                requested_max_tokens=int(kwargs["max_tokens"]),
                response=LLMResponse(content="", finish_reason="length", output_tokens=100),
                stage=str(kwargs.get("stage") or "obsidian_faq"),
            )
        return "## よくある質問\n\n### リリース時期は？\n来週金曜日です。"

    pipeline.llm_service.call_llm = retry_call

    prep = pipeline.prepare_obsidian_export(
        vault,
        meeting_context={
            "raw_notes": "# Multilingual Sync\nBody.",
            "output_language": "Japanese",
        },
        chat_history=[
            {"role": "user", "content": "When is release?"},
            {"role": "assistant", "content": "Next Friday."},
        ],
        base_url="http://localhost:4000",
        api_key="test-key",
        model="o3-mini",
    )
    assert "来週金曜日" in (prep["faq_markdown"] or "")
    assert len(attempts) == 2
    assert attempts[1] > attempts[0]
    sys_prompt = pipeline.llm_service.calls[0]["kwargs"]["system_prompt"]
    assert "Japanese" in sys_prompt

    # Missing base_url/model with valid non-empty chat raises ValueError without Provider calls.
    plain_pipeline = _make_obsidian_pipeline()
    with pytest.raises(ValueError, match="LLM base URL is required"):
        plain_pipeline.prepare_obsidian_export(
            vault,
            raw_notes="# Missing Config\nBody.",
            chat_history=[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}],
            base_url="",
            model="gpt-4o",
        )
    assert plain_pipeline.llm_service.calls == []

    # Tight byte budget shrinks notes excerpt before trimming oldest chat turns.
    monkeypatch.setenv("LLM_MAX_CHAT_REQUEST_BYTES", "1600")
    budget_pipeline = _make_obsidian_pipeline(reply="## FAQ\n\n### Q2?\nA2.")
    long_notes = "# Budget Test\n" + ("LONG_NOTES_EXCERPT " * 200)
    budget_pipeline.prepare_obsidian_export(
        vault,
        raw_notes=long_notes,
        chat_history=[
            {"role": "user", "content": "Q1 old? " + ("x" * 300)},
            {"role": "assistant", "content": "A1 old. " + ("y" * 300)},
            {"role": "user", "content": "Q2 recent?"},
            {"role": "assistant", "content": "A2 recent."},
        ],
        **PROXY_ARGS,
    )
    sent_blob = json.dumps(budget_pipeline.llm_service.calls[0]["messages"])
    assert "Q2 recent?" in sent_blob
    assert "A2 recent." in sent_blob
    assert "LONG_NOTES_EXCERPT" not in sent_blob


def test_export_to_obsidian_faq_distillation_edit_and_stale_cleanup_end_to_end(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline(reply="## Initial FAQ\n\n### Q1\nA1.")
    vault = tmp_path / "vault"
    vault.mkdir()

    prep1 = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Weekly Review\nSummary body.",
        source_stem="weekly",
        chat_history=[{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1."}],
        export_date="2026-11-24",
        **PROXY_ARGS,
    )
    assert prep1["faq_markdown"] == "## Initial FAQ\n\n### Q1\nA1."
    calls_after_prep = len(pipeline.llm_service.calls)

    # User edits FAQ preview before confirming export; no additional Provider calls occur.
    res1 = pipeline.export_to_obsidian(prep1, faq_markdown="## Edited FAQ\n\n### Q1\nFinal A1.")
    assert len(pipeline.llm_service.calls) == calls_after_prep
    tdir = Path(res1["target_dir"])
    faq_file = tdir / prep1["faq_filename"]
    assert faq_file.is_file()
    assert faq_file.read_text(encoding="utf-8").strip() == "## Edited FAQ\n\n### Q1\nFinal A1."
    (tdir / "user_notes.md").write_text("Keep me", encoding="utf-8")

    # Confirmed overwrite when chat now only has placeholder removes stale FAQ while keeping user files.
    prep2 = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Weekly Review\nUpdated summary.",
        source_stem="weekly",
        chat_history=[
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "Please generate meeting notes first before asking questions about the meeting.",
            },
        ],
        export_date="2026-11-24",
    )
    assert prep2["exists"] is True
    assert prep2["faq_markdown"] is None
    res2 = pipeline.export_to_obsidian(prep2, overwrite=True)
    assert res2["faq_path"] is None
    assert not faq_file.exists()
    assert (tdir / "user_notes.md").read_text(encoding="utf-8") == "Keep me"


def test_query_meeting_safe_still_retains_inline_error_history_after_normalization_upgrade(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline()

    def broken_call(messages, **kwargs):
        raise ConnectionError("proxy down")

    pipeline.llm_service.call_llm = broken_call
    ctx = MeetingContext(
        markdown_notes="# Sync\nBody.",
        raw_notes="# Sync\nBody.",
        transcript_evidence="[00:00:05] Hello",
        output_language="English",
    )
    prior_history = [
        ("Earlier question?", "Earlier answer."),
        SimpleNamespace(role="user", content=[{"type": "text", "text": "Multimodal Q?"}]),
        {"role": "assistant", "content": {"type": "text", "text": "Multimodal A."}},
    ]
    reply, updated = pipeline.query_meeting_safe(
        "Current failing question?",
        prior_history,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert "Unable to answer the question right now" in reply
    assert len(updated) == 6
    assert updated[4] == {"role": "user", "content": "Current failing question?"}
    assert updated[5] == {"role": "assistant", "content": reply}


def test_prepare_obsidian_export_filters_real_query_meeting_safe_gr_error_and_exception_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pipeline = _make_obsidian_pipeline(reply="## FAQ\n\n### Q?\nAnswer.")
    vault = tmp_path / "vault"
    vault.mkdir()
    ctx = MeetingContext(
        markdown_notes="# Sync\nNotes.",
        raw_notes="# Sync\nNotes.",
        transcript_evidence="[00:01] Alice spoke.",
        output_language="English",
    )

    # 1. Missing context -> unquoted inline placeholder
    _, hist = pipeline.query_meeting_safe("Failed Q1 missing ctx?", [], None, **PROXY_ARGS)

    # 2. Missing base_url -> gr.Error.__str__ repr()-quoted "'LLM base URL is required.'"
    _, hist = pipeline.query_meeting_safe(
        "Failed Q2 missing base_url?",
        hist,
        ctx,
        "",
        PROXY_ARGS["api_key"],
        PROXY_ARGS["model"],
    )
    assert hist[-1]["content"] == repr("LLM base URL is required.")

    # 3. Missing model -> gr.Error.__str__ repr()-quoted "'Model name is required.'"
    _, hist = pipeline.query_meeting_safe(
        "Failed Q3 missing model?",
        hist,
        ctx,
        PROXY_ARGS["base_url"],
        PROXY_ARGS["api_key"],
        "",
    )
    assert hist[-1]["content"] == repr("Model name is required.")

    # 4. Runtime exception -> "Unable to answer the question right now...\n\nError: RuntimeError: ..."
    original_call_llm = pipeline.llm_service.call_llm

    def raising_call(messages, **kwargs):
        raise RuntimeError('It\'s "proxy" down')

    monkeypatch.setattr(pipeline.llm_service, "call_llm", raising_call)
    _, hist = pipeline.query_meeting_safe(
        "Failed Q4 runtime err?",
        hist,
        ctx,
        **PROXY_ARGS,
    )
    monkeypatch.setattr(pipeline.llm_service, "call_llm", original_call_llm)

    # 5. Direct gr.Error with escaped quotes and newline inside repr()
    hist_with_escaped = [
        *hist,
        {"role": "user", "content": "Failed Q5 multiline repr?"},
        {
            "role": "assistant",
            "content": str(gr.Error('Error: ConnectionError: It\'s "refused"\nRetry')),
        },
        {"role": "user", "content": "Failed Q6 blank repr?"},
        {"role": "assistant", "content": str(gr.Error("  \n\t  "))},
    ]

    pipeline.llm_service.calls.clear()
    prep_errors = pipeline.prepare_obsidian_export(
        vault,
        meeting_context=ctx,
        chat_history=hist_with_escaped,
        **PROXY_ARGS,
    )
    assert prep_errors["faq_markdown"] is None
    assert pipeline.llm_service.calls == []

    # Recovery + quoted-prose non-false-positives (must NOT be filtered out)
    hist_recovered = [
        *hist_with_escaped,
        {"role": "user", "content": "What did Alice say?"},
        {
            "role": "assistant",
            "content": 'Alice said "LLM base URL is required" in the meeting.',
        },
        {"role": "user", "content": "What was on slide 3?"},
        {
            "role": "assistant",
            "content": '"LLM base URL is required" was shown on slide 3.',
        },
        {"role": "user", "content": "What did Bob quote?"},
        {
            "role": "assistant",
            "content": '"LLM base URL is required" was quoted by "Bob"',
        },
    ]
    prep_recovered = pipeline.prepare_obsidian_export(
        vault,
        meeting_context=ctx,
        chat_history=hist_recovered,
        **PROXY_ARGS,
    )
    assert prep_recovered["faq_markdown"] == "## FAQ\n\n### Q?\nAnswer."
    assert len(pipeline.llm_service.calls) == 1
    call = pipeline.llm_service.calls[0]
    assert call["kwargs"]["stage"] == "obsidian_faq"
    prompt_blob = json.dumps(call["messages"])
    assert 'Alice said \\"LLM base URL is required\\" in the meeting.' in prompt_blob
    assert '\\"LLM base URL is required\\" was shown on slide 3.' in prompt_blob
    assert '\\"LLM base URL is required\\" was quoted by \\"Bob\\"' in prompt_blob
    for dropped in (
        "Failed Q1 missing ctx?",
        "Failed Q2 missing base_url?",
        "Failed Q3 missing model?",
        "Failed Q4 runtime err?",
        "Failed Q5 multiline repr?",
        "Failed Q6 blank repr?",
        "Unable to answer the question right now",
        "ConnectionError",
    ):
        assert dropped not in prompt_blob


def test_prepare_obsidian_export_rejects_malformed_chat_before_title_llm_without_h1(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline(reply="Should not be called")
    vault = tmp_path / "vault"
    vault.mkdir()

    # No H1 heading: _derive_obsidian_title would call Provider if chat_history wasn't validated first.
    with pytest.raises(ValueError, match="chat_history must be a JSON list"):
        pipeline.prepare_obsidian_export(
            vault,
            raw_notes="## Summary\nNo H1 heading here.",
            source_stem="no_h1",
            chat_history="not-json",
            **PROXY_ARGS,
        )
    assert pipeline.llm_service.calls == []

    # Invalid vault_path and empty raw_notes still raise before chat_history validation.
    with pytest.raises(ValueError, match="Obsidian vault folder does not exist"):
        pipeline.prepare_obsidian_export(
            tmp_path / "missing_vault",
            raw_notes="## Summary\nNo H1 heading here.",
            chat_history="not-json",
            **PROXY_ARGS,
        )
    with pytest.raises(ValueError, match="No Meeting Notes available"):
        pipeline.prepare_obsidian_export(
            vault,
            raw_notes="   ",
            chat_history="not-json",
            **PROXY_ARGS,
        )
    assert pipeline.llm_service.calls == []


def test_prepare_obsidian_export_preserves_qa_pair_and_trailing_assistant_under_tiny_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LLM_MAX_CHAT_REQUEST_BYTES", "1500")
    vault = tmp_path / "vault"
    vault.mkdir()

    # 1. Single oversized [user, assistant] pair must keep both turns intact.
    pair_pipeline = _make_obsidian_pipeline(reply="## FAQ\n\n### FINAL_Q\nFINAL_A")
    pair_pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Tiny Budget Pair\nNotes.",
        chat_history=[
            {"role": "user", "content": "FINAL_Q"},
            {"role": "assistant", "content": "FINAL_A " + ("pad " * 120)},
        ],
        **PROXY_ARGS,
    )
    assert len(pair_pipeline.llm_service.calls) == 1
    pair_blob = json.dumps(pair_pipeline.llm_service.calls[0]["messages"])
    assert "FINAL_Q" in pair_blob
    assert "FINAL_A" in pair_blob

    # 2. Trailing assistant continuation [U1, A1, A2] must keep U1, A1, and A2 intact.
    cont_pipeline = _make_obsidian_pipeline(reply="## FAQ\n\n### U1_Q\nA1_ANS")
    cont_pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Tiny Budget Continuation\nNotes.",
        chat_history=[
            {"role": "user", "content": "U1_Q"},
            {"role": "assistant", "content": "A1_ANS"},
            {"role": "assistant", "content": "A2_CONT " + ("pad " * 120)},
        ],
        **PROXY_ARGS,
    )
    assert len(cont_pipeline.llm_service.calls) == 1
    cont_blob = json.dumps(cont_pipeline.llm_service.calls[0]["messages"])
    assert "U1_Q" in cont_blob
    assert "A1_ANS" in cont_blob
    assert "A2_CONT" in cont_blob


def test_prepare_obsidian_export_extracts_attribute_object_fallbacks_in_chat_history(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline(reply="## FAQ\n\n### Obj Q\nObj A")
    vault = tmp_path / "vault"
    vault.mkdir()

    chat_history = [
        SimpleNamespace(
            role="user",
            content=SimpleNamespace(text="", content="Obj Q fallback"),
        ),
        SimpleNamespace(
            role="assistant",
            content=[
                SimpleNamespace(text="", content="Obj A fallback"),
                SimpleNamespace(text="   ", value="Obj V fallback"),
            ],
        ),
    ]
    prepared = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Attr Objects\nBody.",
        chat_history=chat_history,
        **PROXY_ARGS,
    )
    assert prepared["faq_markdown"] == "## FAQ\n\n### Obj Q\nObj A"
    assert len(pipeline.llm_service.calls) == 1
    prompt_blob = json.dumps(pipeline.llm_service.calls[0]["messages"])
    assert "Obj Q fallback" in prompt_blob
    assert "Obj A fallback" in prompt_blob
    assert "Obj V fallback" in prompt_blob


def test_export_to_obsidian_cleanly_removes_stale_faq_directory_on_overwrite(
    tmp_path: Path,
):
    pipeline = _make_obsidian_pipeline()
    vault = tmp_path / "vault"
    vault.mkdir()

    target_dir = vault / "2026-11-24 Dir FAQ Test"
    target_dir.mkdir()
    faq_dir = target_dir / "2026-11-24 Dir FAQ Test - FAQ.md"
    faq_dir.mkdir()
    (faq_dir / "nested.txt").write_text("stale nested", encoding="utf-8")
    (target_dir / "keep_me.md").write_text("User note", encoding="utf-8")

    # 1. Overwrite with empty FAQ removes directory at faq_path without PermissionError/IsADirectoryError.
    prep_empty = pipeline.prepare_obsidian_export(
        vault,
        raw_notes="# Dir FAQ Test\nUpdated body.",
        export_date="2026-11-24",
    )
    assert prep_empty["exists"] is True
    assert prep_empty["faq_markdown"] is None
    res1 = pipeline.export_to_obsidian(prep_empty, overwrite=True)
    assert res1["faq_path"] is None
    assert not faq_dir.exists()
    assert (target_dir / "keep_me.md").read_text(encoding="utf-8") == "User note"

    # 2. Overwrite when faq_path is a directory and faq_markdown is non-empty replaces it cleanly.
    faq_dir.mkdir()
    (faq_dir / "nested.txt").write_text("stale nested 2", encoding="utf-8")
    res2 = pipeline.export_to_obsidian(
        prep_empty,
        faq_markdown="## Replaced FAQ\n\n### Q?\nA.",
        overwrite=True,
    )
    assert res2["faq_path"] == str(faq_dir)
    assert faq_dir.is_file()
    assert faq_dir.read_text(encoding="utf-8").strip() == "## Replaced FAQ\n\n### Q?\nA."
    assert (target_dir / "keep_me.md").read_text(encoding="utf-8") == "User note"



