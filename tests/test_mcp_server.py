"""Tests for the MCP server tools and resources.

The MCP server instantiates service singletons at module level, so these tests
mock the module-level service objects (``_llm``, ``_video``, ``_notes``,
``_pipeline``) and the ``_resolve_llm_params`` helper to avoid needing real
LLM connections or video files.

All filesystem-dependent tests use ``tmp_path`` and ``monkeypatch`` for
complete isolation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the MCP server module.  Because mcp_server.py calls
# setup_runtime_environment() and instantiates services at import time,
# we patch those away so the tests don't depend on .env or real services.
# ---------------------------------------------------------------------------
import mcp_server


# ===================================================================
# Helpers
# ===================================================================

def _make_minutes_file(directory: Path, name: str, content: str = "# Meeting Notes\nSample") -> Path:
    """Create a .minutes.md file in the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


# ===================================================================
# _resolve_llm_params
# ===================================================================


class TestResolveLlmParams:
    """Tests for the _resolve_llm_params helper function."""

    def test_explicit_values_override_env(self):
        result = mcp_server._resolve_llm_params(
            base_url="http://custom:9999",
            api_key="my-key",
            model="gpt-4o-mini",
        )
        assert result["base_url"] == "http://custom:9999"
        assert result["api_key"] == "my-key"
        assert result["model"] == "gpt-4o-mini"
        assert set(result) == {"base_url", "api_key", "model"}

    def test_falls_back_to_env_defaults(self):
        mock_defaults = MagicMock()
        mock_defaults.base_url = "http://env-url:4000"
        mock_defaults.api_key = "env-key"
        mock_defaults.model = "gpt-4o"

        with patch("mcp_server.ProxyDefaults.from_env", return_value=mock_defaults):
            result = mcp_server._resolve_llm_params()

        assert result["base_url"] == "http://env-url:4000"
        assert result["api_key"] == "env-key"
        assert result["model"] == "gpt-4o"

    def test_empty_strings_fall_back_to_env(self):
        mock_defaults = MagicMock()
        mock_defaults.base_url = "http://fallback"
        mock_defaults.api_key = "fallback-key"
        mock_defaults.model = "fallback-model"

        with patch("mcp_server.ProxyDefaults.from_env", return_value=mock_defaults):
            result = mcp_server._resolve_llm_params(
                base_url="", api_key="  ", model="",
            )

        assert result["base_url"] == "http://fallback"
        assert result["api_key"] == "fallback-key"


# ===================================================================
# test_llm_connection tool
# ===================================================================


class TestTestLlmConnection:
    """Tests for the test_llm_connection MCP tool."""

    def test_delegates_to_service(self):
        with patch.object(mcp_server, "_llm") as mock_llm, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {
                "base_url": "http://test",
                "api_key": "key",
                "model": "gpt-4o",
            }
            mock_llm.test_connection.return_value = "✅ Connected"

            result = mcp_server.test_llm_connection()

        mock_llm.test_connection.assert_called_once_with(
            base_url="http://test", api_key="key", model="gpt-4o",
        )
        assert result == "✅ Connected"

    def test_with_explicit_overrides(self):
        with patch.object(mcp_server, "_llm") as mock_llm, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {
                "base_url": "http://override",
                "api_key": "override-key",
                "model": "custom-model",
            }
            mock_llm.test_connection.return_value = "✅ OK"

            result = mcp_server.test_llm_connection(
                base_url="http://override", api_key="override-key", model="custom-model",
            )

        assert result == "✅ OK"
        mock_resolve.assert_called_once()


# ===================================================================
# generate_meeting_notes tool
# ===================================================================


class TestGenerateMeetingNotes:
    """Tests for the generate_meeting_notes MCP tool."""

    def test_no_input_returns_error(self):
        result = mcp_server.generate_meeting_notes()
        assert "Error" in result
        assert "video_path" in result or "transcript_path" in result

    def test_with_video_path_success(self, tmp_path):
        mock_pipeline_result = (
            [{"thumb": "img1"}, {"thumb": "img2"}],  # thumbnails
            "# Notes\nSample markdown",               # markdown_notes
            "<h1>Notes</h1>",                          # html_notes
            str(tmp_path / "report.html"),             # html_path
        )

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._latest_minutes_path") as mock_latest, \
             patch("mcp_server._resolve_llm_params") as mock_resolve, \
             patch("gradio.Progress", return_value=MagicMock()):
            mock_resolve.return_value = {
                "base_url": "http://test", "api_key": "k", "model": "m",
            }
            mock_pipeline.run_pipeline.return_value = mock_pipeline_result
            mock_latest.return_value = tmp_path / "minutes.md"

            result = mcp_server.generate_meeting_notes(video_path="/fake/video.mp4")

        data = json.loads(result)
        assert data["markdown_notes"] == "# Notes\nSample markdown"
        assert data["screenshots_count"] == 2
        assert "html_path" in data
        assert "minutes_path" in data

    def test_with_transcript_path_success(self, tmp_path):
        mock_pipeline_result = (
            [],                                  # no thumbnails
            "# Transcript Notes",                # markdown
            "<h1>Transcript Notes</h1>",         # html
            str(tmp_path / "report.html"),       # html_path
        )

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._latest_minutes_path") as mock_latest, \
             patch("mcp_server._resolve_llm_params") as mock_resolve, \
             patch("gradio.Progress", return_value=MagicMock()):
            mock_resolve.return_value = {
                "base_url": "http://test", "api_key": "k", "model": "m",
            }
            mock_pipeline.run_pipeline.return_value = mock_pipeline_result
            mock_latest.return_value = None

            result = mcp_server.generate_meeting_notes(transcript_path="/fake/transcript.txt")

        data = json.loads(result)
        assert data["markdown_notes"] == "# Transcript Notes"
        assert data["screenshots_count"] == 0
        assert data["minutes_path"] == ""

    def test_pipeline_exception_returns_error(self):
        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve, \
             patch("gradio.Progress", return_value=MagicMock()):
            mock_resolve.return_value = {
                "base_url": "http://test", "api_key": "k", "model": "m",
            }
            mock_pipeline.run_pipeline.side_effect = RuntimeError("LLM timeout")

            result = mcp_server.generate_meeting_notes(video_path="/fake/video.mp4")

        assert "Error" in result
        assert "RuntimeError" in result
        assert "LLM timeout" in result

    def test_none_thumbnails_returns_zero_screenshots(self, tmp_path):
        mock_pipeline_result = (None, "notes", "html", str(tmp_path / "r.html"))

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._latest_minutes_path") as mock_latest, \
             patch("mcp_server._resolve_llm_params") as mock_resolve, \
             patch("gradio.Progress", return_value=MagicMock()):
            mock_resolve.return_value = {
                "base_url": "u", "api_key": "k", "model": "m",
            }
            mock_pipeline.run_pipeline.return_value = mock_pipeline_result
            mock_latest.return_value = None

            result = mcp_server.generate_meeting_notes(video_path="/fake/v.mp4")

        data = json.loads(result)
        assert data["screenshots_count"] == 0


# ===================================================================
# list_meeting_minutes tool
# ===================================================================


class TestListMeetingMinutes:
    """Tests for the list_meeting_minutes MCP tool."""

    def test_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = mcp_server.list_meeting_minutes()
        assert json.loads(result) == []

    def test_no_minutes_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = mcp_server.list_meeting_minutes()
        entries = json.loads(result)
        assert entries == []

    def test_with_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        _make_minutes_file(minutes_dir, "meeting1.minutes.md", "# Meeting 1")
        _make_minutes_file(minutes_dir, "meeting2.minutes.md", "# Meeting 2")
        # Non-minutes file should be excluded
        (minutes_dir / "notes.txt").write_text("not a minutes file")

        result = mcp_server.list_meeting_minutes()
        entries = json.loads(result)

        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "meeting1.minutes.md" in names
        assert "meeting2.minutes.md" in names

        for entry in entries:
            assert "path" in entry
            assert "size_kb" in entry
            assert "modified_at" in entry
            assert isinstance(entry["size_kb"], float)

    def test_sorted_by_mtime_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"

        older = _make_minutes_file(minutes_dir, "older.minutes.md", "old")
        time.sleep(0.05)
        newer = _make_minutes_file(minutes_dir, "newer.minutes.md", "new")

        result = mcp_server.list_meeting_minutes()
        entries = json.loads(result)

        assert len(entries) == 2
        assert entries[0]["name"] == "newer.minutes.md"
        assert entries[1]["name"] == "older.minutes.md"


# ===================================================================
# get_meeting_minutes tool
# ===================================================================


class TestGetMeetingMinutes:
    """Tests for the get_meeting_minutes MCP tool."""

    def test_absolute_path(self, tmp_path):
        content = "# Absolute Path Meeting\nDetails here."
        path = _make_minutes_file(tmp_path, "abs.minutes.md", content)

        result = mcp_server.get_meeting_minutes(str(path))
        assert result == content

    def test_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        content = "# Relative Path Meeting"
        _make_minutes_file(minutes_dir, "rel.minutes.md", content)

        result = mcp_server.get_meeting_minutes("rel.minutes.md")
        assert result == content

    def test_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = mcp_server.get_meeting_minutes("nonexistent.minutes.md")
        assert "Error" in result
        assert "not found" in result


# ===================================================================
# get_latest_meeting_minutes tool
# ===================================================================


class TestGetLatestMeetingMinutes:
    """Tests for the get_latest_meeting_minutes MCP tool."""

    def test_found(self, tmp_path):
        content = "# Latest Meeting Notes\nImportant decisions."
        path = _make_minutes_file(tmp_path, "latest.minutes.md", content)

        with patch("mcp_server._latest_minutes_path", return_value=path):
            result = mcp_server.get_latest_meeting_minutes()

        assert result == content

    def test_none_found(self):
        with patch("mcp_server._latest_minutes_path", return_value=None):
            result = mcp_server.get_latest_meeting_minutes()

        assert result == "No meeting minutes found."


# ===================================================================
# list_note_styles tool
# ===================================================================


class TestListNoteStyles:
    """Tests for the list_note_styles MCP tool."""

    def test_returns_valid_json(self):
        with patch.object(mcp_server, "_notes") as mock_notes:
            mock_notes.STYLE_GUIDES = {
                "Executive Summary": "Concise summary.",
                "Detailed Notes": "Comprehensive notes.",
                "Action Items Focus": "Action items and owners.",
                "Technical Minutes": "Formal minutes.",
            }
            result = mcp_server.list_note_styles()

        data = json.loads(result)
        assert isinstance(data, dict)
        assert "Executive Summary" in data
        assert "Detailed Notes" in data
        assert "Action Items Focus" in data
        assert "Technical Minutes" in data

    def test_matches_notes_service_styles(self):
        """Ensure the tool returns what the underlying service provides."""
        from meeting_notes.notes import NotesService
        expected = NotesService.STYLE_GUIDES

        with patch.object(mcp_server, "_notes") as mock_notes:
            mock_notes.STYLE_GUIDES = expected
            result = mcp_server.list_note_styles()

        data = json.loads(result)
        assert data == expected


# ===================================================================
# get_video_info tool
# ===================================================================


class TestGetVideoInfo:
    """Tests for the get_video_info MCP tool."""

    def test_success(self):
        info = {"duration": 120.5, "resolution": "1920x1080", "size_mb": 45.2}

        with patch.object(mcp_server, "_video") as mock_video:
            mock_video.get_video_info.return_value = info
            result = mcp_server.get_video_info("/path/to/video.mp4")

        data = json.loads(result)
        assert data["duration"] == 120.5
        assert data["resolution"] == "1920x1080"
        assert data["size_mb"] == 45.2

    def test_error(self):
        with patch.object(mcp_server, "_video") as mock_video:
            mock_video.get_video_info.side_effect = ValueError("Cannot open video")
            result = mcp_server.get_video_info("/bad/path.mp4")

        assert "Error" in result
        assert "ValueError" in result
        assert "Cannot open video" in result

    def test_file_not_found_error(self):
        with patch.object(mcp_server, "_video") as mock_video:
            mock_video.get_video_info.side_effect = FileNotFoundError("No such file")
            result = mcp_server.get_video_info("/missing/video.mp4")

        assert "Error" in result
        assert "FileNotFoundError" in result


# ===================================================================
# MCP Resources
# ===================================================================


class TestResourceLatestMinutes:
    """Tests for the meeting-notes://minutes/latest resource."""

    def test_found(self, tmp_path):
        content = "# Resource Latest\nResource content."
        path = _make_minutes_file(tmp_path, "resource.minutes.md", content)

        with patch("mcp_server._latest_minutes_path", return_value=path):
            result = mcp_server.resource_latest_minutes()

        assert result == content

    def test_not_found(self):
        with patch("mcp_server._latest_minutes_path", return_value=None):
            result = mcp_server.resource_latest_minutes()

        assert result == "No meeting minutes found yet."


class TestResourceMinutesList:
    """Tests for the meeting-notes://minutes/list resource."""

    def test_delegates_to_list_meeting_minutes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        _make_minutes_file(minutes_dir, "test.minutes.md", "# Test")

        result = mcp_server.resource_minutes_list()
        entries = json.loads(result)
        assert len(entries) == 1
        assert entries[0]["name"] == "test.minutes.md"

    def test_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = mcp_server.resource_minutes_list()
        assert json.loads(result) == []


class TestResourceConfigStatus:
    """Tests for the meeting-notes://config/status resource."""

    def test_returns_markdown(self):
        mock_defaults = MagicMock()
        mock_defaults.status_markdown.return_value = "Loaded defaults:\n- Base URL: http://test"

        with patch("mcp_server.ProxyDefaults.from_env", return_value=mock_defaults):
            result = mcp_server.resource_config_status()

        assert "Loaded defaults" in result
        assert "Base URL" in result
        mock_defaults.status_markdown.assert_called_once()


# ===================================================================
# Obsidian Export Helpers & Tools (Ticket 05)
# ===================================================================


def _make_fake_pipeline(faq_reply: str = "## Distilled FAQ\n\n### Q?\nA."):
    from meeting_notes.notes import NotesService
    from meeting_notes.pipeline import PipelineService
    from tests.conftest import FakeLLMService, FakeVideoService

    class _CountingFakeLLM(FakeLLMService):
        def __init__(self):
            super().__init__(reply=faq_reply)

    llm = _CountingFakeLLM()
    video = FakeVideoService(llm)
    notes = NotesService()
    return PipelineService(llm, video, notes), llm


def _create_dummy_screenshots(directory: Path, count: int = 2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        (directory / f"screenshot_{idx:02d}.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9")
    return directory


class TestExtractCleanMinutesBody:
    """Tests for `.minutes.md` parsing, source stem, and Screenshot dir resolution."""

    def test_strips_meeting_minutes_header_and_metadata_only_at_top(self):
        raw = (
            "\ufeff# Meeting Minutes\n\n"
            "source_video: product_kickoff.mp4\n"
            "transcript_file: product_kickoff.transcript.txt\n"
            "transcript_manifest: product_kickoff.manifest.json\n"
            "ocr_available: True\n"
            "screenshot_interval_seconds: 30\n"
            "max_screenshots: 8\n"
            "screenshots_captured: 2\n"
            "generated_at: 2026-11-22T10:00:00\n\n"
            "# Product Kickoff\n\n"
            "Note: keep this prose line starting with Note:.\n"
            "Summary: also keep this line.\n"
        )
        body, metadata = mcp_server._parse_minutes_document(raw)

        assert "# Meeting Minutes" not in body
        assert "source_video:" not in body
        assert "generated_at:" not in body
        assert body.startswith("# Product Kickoff")
        assert "Note: keep this prose line starting with Note:." in body
        assert "Summary: also keep this line." in body
        assert metadata["source_video"] == "product_kickoff.mp4"
        assert metadata["screenshots_captured"] == "2"

    def test_strips_trailing_alignment_checks_but_keeps_fenced_and_mid_doc(self):
        raw = (
            "# Meeting Minutes\n\nsource_video: sync.mp4\n\n"
            "# Weekly Sync\n\n"
            "```markdown\n"
            "## Screenshot Alignment Checks\n"
            "- Example inside code fence\n"
            "```\n\n"
            "## Screenshot Alignment Checks\n\n"
            "- Mid-document warning before real heading\n\n"
            "## Action Items\n\n- Ship feature\n\n"
            "## Screenshot Alignment Checks\n\n"
            "- Screenshot 1 at 0:02 may be weakly aligned.\n"
        )
        clean = mcp_server._extract_clean_minutes_body(raw)

        assert "Example inside code fence" in clean
        assert "## Action Items" in clean
        assert "- Ship feature" in clean
        assert "Mid-document warning before real heading" in clean
        assert "Screenshot 1 at 0:02 may be weakly aligned." not in clean
        assert not clean.endswith("## Screenshot Alignment Checks")

    def test_preserves_plain_markdown_and_headerless_subheadings(self):
        plain = "# Already Plain\n\nPlain body."
        assert mcp_server._extract_clean_minutes_body(plain) == plain

        no_h1_minutes = (
            "# Meeting Minutes\n\n"
            "source_transcript: ops.txt\n\n"
            "## Summary\nNo H1 heading here.\n"
        )
        extracted = mcp_server._extract_clean_minutes_body(no_h1_minutes)
        assert extracted == "## Summary\nNo H1 heading here."

    def test_extract_source_stem_prefers_metadata_and_strips_minutes_suffix(self, tmp_path):
        mf = tmp_path / "weekly_ops.minutes.md"
        assert mcp_server._extract_minutes_source_stem(
            mf, {"source_transcript": "ops_transcript_stem.txt"}
        ) == "ops_transcript_stem"
        assert mcp_server._extract_minutes_source_stem(mf, {}) == "weekly_ops"

    def test_resolve_screenshots_dir_handles_url_encoded_traversal_and_transcript_only(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        minutes_dir.mkdir(parents=True)
        shot_dir = _create_dummy_screenshots(minutes_dir / "images" / "team sync", count=1)
        mf = minutes_dir / "team sync.minutes.md"

        resolved = mcp_server._resolve_minutes_screenshots_dir(
            mf,
            "![Shot](images/team%20sync/screenshot_01.jpg)",
            {"source_video": "team sync.mp4"},
            "team sync",
        )
        assert resolved == str(shot_dir)

        # Path traversal in link stem must be ignored
        outside = tmp_path / "secret"
        _create_dummy_screenshots(outside, count=1)
        assert mcp_server._resolve_minutes_screenshots_dir(
            mf,
            "![Bad](images/..%2F..%2Fsecret/screenshot_01.jpg)",
            {"source_transcript": "only_tx.txt"},
            "only_tx",
        ) is None

        # Transcript-only without video or links returns None even if matching dir exists
        _create_dummy_screenshots(minutes_dir / "images" / "only_tx", count=1)
        assert mcp_server._resolve_minutes_screenshots_dir(
            minutes_dir / "only_tx.minutes.md",
            "# Tx Only\nText.",
            {"source_transcript": "only_tx.txt"},
            "only_tx",
        ) is None


class TestPrepareObsidianExport:
    """Tests for the stateless prepare_obsidian_export MCP tool."""

    def test_tool_is_registered(self):
        assert "prepare_obsidian_export" in mcp_server.mcp._tool_manager._tools

    def test_missing_inputs_returns_error_without_calling_pipeline(self, tmp_path):
        with patch.object(mcp_server, "_pipeline") as mock_pipeline:
            res_empty = mcp_server.prepare_obsidian_export(vault_path=str(tmp_path))
            res_blank = mcp_server.prepare_obsidian_export(
                vault_path=str(tmp_path), minutes_file="  ", notes_markdown=" \n "
            )

        assert res_empty.startswith("Error:")
        assert res_blank.startswith("Error:")
        mock_pipeline.prepare_obsidian_export.assert_not_called()

    def test_nonexistent_minutes_file_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch.object(mcp_server, "_pipeline") as mock_pipeline:
            res = mcp_server.prepare_obsidian_export(
                vault_path=str(tmp_path), minutes_file="missing.minutes.md"
            )

        assert "Error:" in res
        assert "not found" in res
        mock_pipeline.prepare_obsidian_export.assert_not_called()

    def test_delegates_relative_minutes_file_and_llm_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        shots = _create_dummy_screenshots(minutes_dir / "images" / "sync_stem", count=2)
        _make_minutes_file(
            minutes_dir,
            "sync_stem.minutes.md",
            (
                "# Meeting Minutes\n\n"
                "source_video: sync_stem.mp4\n"
                "generated_at: 2026-11-22T10:00:00\n\n"
                "# Weekly Sync\n\nBody.\n\n"
                "## Screenshot Alignment Checks\n\n- Warning 1\n"
            ),
        )
        vault = tmp_path / "vault"
        vault.mkdir()

        expected = {
            "vault_path": str(vault),
            "folder_name": "2026-11-24 Weekly Sync",
            "target_dir": str(vault / "2026-11-24 Weekly Sync"),
            "notes_filename": "2026-11-24 Weekly Sync.md",
            "faq_filename": "2026-11-24 Weekly Sync - FAQ.md",
            "exists": False,
            "faq_markdown": None,
            "screenshot_count": 2,
            "default_attachment_prefix": "screenshot",
        }
        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve, \
             patch("mcp_server._default_reasoning_effort", return_value="medium"):
            mock_resolve.return_value = {
                "base_url": "http://proxy:4000",
                "api_key": "k",
                "model": "gpt-4o",
            }
            mock_pipeline.prepare_obsidian_export.return_value = expected

            raw = mcp_server.prepare_obsidian_export(
                vault_path=str(vault),
                minutes_file="sync_stem.minutes.md",
                attachment_prefix="wk",
            )

        assert json.loads(raw) == expected
        mock_pipeline.prepare_obsidian_export.assert_called_once_with(
            vault_path=str(vault),
            notes_markdown="# Weekly Sync\n\nBody.",
            source_stem="sync_stem",
            screenshots_dir=str(shots),
            chat_history=None,
            base_url="http://proxy:4000",
            api_key="k",
            model="gpt-4o",
            output_language=None,
            reasoning_effort="medium",
            export_date=None,
            attachment_prefix="wk",
        )

    def test_pipeline_exception_returns_formatted_error(self, tmp_path):
        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {"base_url": "", "api_key": "", "model": ""}
            mock_pipeline.prepare_obsidian_export.side_effect = ValueError("Invalid vault")

            raw = mcp_server.prepare_obsidian_export(
                vault_path=str(tmp_path / "missing_vault"),
                notes_markdown="# Title\nBody.",
            )

        assert raw == "Error: ValueError: Invalid vault"

    def test_seam_distills_chat_history_from_minutes_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pipeline, llm = _make_fake_pipeline(faq_reply="## FAQ\n\n### Owner?\nAlice.")
        monkeypatch.setattr(mcp_server, "_pipeline", pipeline)
        monkeypatch.setattr(mcp_server, "_llm", llm)

        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        _create_dummy_screenshots(minutes_dir / "images" / "demo", count=2)
        _make_minutes_file(
            minutes_dir,
            "demo.minutes.md",
            "# Meeting Minutes\n\nsource_video: demo.mp4\n\n# Demo Sync\n\nNotes body.\n",
        )
        vault = tmp_path / "vault"
        vault.mkdir()

        raw = mcp_server.prepare_obsidian_export(
            vault_path=str(vault),
            minutes_file="demo.minutes.md",
            chat_history=json.dumps([
                {"role": "user", "content": "Who owns it?"},
                {"role": "assistant", "content": "Alice."},
            ]),
            base_url="http://localhost:4000",
            api_key="test-key",
            model="gpt-4o",
        )
        data = json.loads(raw)
        assert data["screenshot_count"] == 2
        assert data["exists"] is False
        assert "Alice" in data["faq_markdown"]
        assert len(llm.calls) == 1


class TestExportToObsidian:
    """Tests for the stateless export_to_obsidian MCP tool."""

    def test_tool_is_registered(self):
        assert "export_to_obsidian" in mcp_server.mcp._tool_manager._tools

    def test_missing_inputs_and_missing_file_return_errors(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert mcp_server.export_to_obsidian(vault_path=str(tmp_path)).startswith("Error:")
        res = mcp_server.export_to_obsidian(
            vault_path=str(tmp_path), minutes_file="absent.minutes.md"
        )
        assert "Error:" in res and "not found" in res

    def test_faq_markdown_override_skips_chat_distillation(self, tmp_path):
        prepared = {"vault_path": str(tmp_path), "folder_name": "2026-11-24 Note"}
        exported = {"status": "exported", "written": True, "exported": True}

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {"base_url": "u", "api_key": "k", "model": "m"}
            mock_pipeline.prepare_obsidian_export.return_value = prepared
            mock_pipeline.export_to_obsidian.return_value = exported

            raw = mcp_server.export_to_obsidian(
                vault_path=str(tmp_path),
                notes_markdown="# Note",
                chat_history=[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}],
                faq_markdown="## Custom FAQ",
                attachment_prefix="shot",
                overwrite=True,
            )

        assert json.loads(raw) == exported
        assert mock_pipeline.prepare_obsidian_export.call_args.kwargs["chat_history"] is None
        mock_pipeline.export_to_obsidian.assert_called_once_with(
            prepared,
            attachment_prefix="shot",
            overwrite=True,
            faq_markdown="## Custom FAQ",
        )

    def test_empty_faq_markdown_suppresses_faq_and_skips_chat_distillation(self, tmp_path):
        prepared = {"vault_path": str(tmp_path), "folder_name": "2026-11-24 Note"}
        exported = {"status": "exported", "written": True, "exported": True, "faq_path": None}

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {"base_url": "u", "api_key": "k", "model": "m"}
            mock_pipeline.prepare_obsidian_export.return_value = prepared
            mock_pipeline.export_to_obsidian.return_value = exported

            raw = mcp_server.export_to_obsidian(
                vault_path=str(tmp_path),
                notes_markdown="# Note",
                chat_history=[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}],
                faq_markdown="",
            )

        assert json.loads(raw) == exported
        assert mock_pipeline.prepare_obsidian_export.call_args.kwargs["chat_history"] is None
        assert mock_pipeline.export_to_obsidian.call_args.kwargs["faq_markdown"] == ""

    def test_chat_history_without_faq_markdown_distills_via_prepare(self, tmp_path):
        prepared = {
            "vault_path": str(tmp_path),
            "folder_name": "2026-11-24 Note",
            "faq_markdown": "## Distilled",
        }
        exported = {"status": "exported", "written": True, "exported": True}
        chat = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {"base_url": "u", "api_key": "k", "model": "m"}
            mock_pipeline.prepare_obsidian_export.return_value = prepared
            mock_pipeline.export_to_obsidian.return_value = exported

            mcp_server.export_to_obsidian(
                vault_path=str(tmp_path),
                notes_markdown="# Note",
                chat_history=chat,
                output_language="Spanish",
                reasoning_effort="low",
            )

        prep_kwargs = mock_pipeline.prepare_obsidian_export.call_args.kwargs
        assert prep_kwargs["chat_history"] == chat
        assert prep_kwargs["output_language"] == "Spanish"
        assert prep_kwargs["reasoning_effort"] == "low"
        assert "faq_markdown" not in mock_pipeline.export_to_obsidian.call_args.kwargs

    def test_explicit_screenshots_dir_overrides_minutes_auto_discovery(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        _create_dummy_screenshots(minutes_dir / "images" / "auto_stem", count=1)
        _make_minutes_file(
            minutes_dir,
            "auto_stem.minutes.md",
            "# Meeting Minutes\n\nsource_video: auto_stem.mp4\n\n# Auto\nBody.",
        )
        custom_shots = _create_dummy_screenshots(tmp_path / "custom_shots", count=3)

        with patch.object(mcp_server, "_pipeline") as mock_pipeline, \
             patch("mcp_server._resolve_llm_params") as mock_resolve:
            mock_resolve.return_value = {"base_url": "", "api_key": "", "model": ""}
            mock_pipeline.prepare_obsidian_export.return_value = {}
            mock_pipeline.export_to_obsidian.return_value = {"status": "exported"}

            mcp_server.export_to_obsidian(
                vault_path=str(tmp_path),
                minutes_file="auto_stem.minutes.md",
                screenshots_dir=str(custom_shots),
            )

        assert (
            mock_pipeline.prepare_obsidian_export.call_args.kwargs["screenshots_dir"]
            == str(custom_shots)
        )

    def test_seam_minutes_export_rewrites_links_distills_faq_and_protects_collisions(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        pipeline, llm = _make_fake_pipeline(faq_reply="## FAQ\n\n### Timeline?\nFriday.")
        monkeypatch.setattr(mcp_server, "_pipeline", pipeline)
        monkeypatch.setattr(mcp_server, "_llm", llm)

        minutes_dir = tmp_path / "artifacts" / "meeting_minutes"
        _create_dummy_screenshots(minutes_dir / "images" / "team sync", count=1)
        mf = _make_minutes_file(
            minutes_dir,
            "team sync.minutes.md",
            (
                "# Meeting Minutes\n\n"
                "source_video: team sync.mp4\n"
                "generated_at: 2026-11-22T12:00:00\n\n"
                "# Team Sync\n\n"
                "![Shot](images/team%20sync/screenshot_01.jpg)\n\n"
                "## Screenshot Alignment Checks\n\n"
                "- Weak alignment warning\n"
            ),
        )
        vault = tmp_path / "vault"
        vault.mkdir()

        # 1. Initial export from .minutes.md + chat_history
        res1 = json.loads(
            mcp_server.export_to_obsidian(
                vault_path=str(vault),
                minutes_file=str(mf),
                chat_history=[
                    {"role": "user", "content": "Timeline?"},
                    {"role": "assistant", "content": "Friday."},
                ],
                attachment_prefix="my shot",
                base_url="http://localhost:4000",
                api_key="test-key",
                model="gpt-4o",
            )
        )
        tdir = Path(res1["target_dir"])
        notes_file = tdir / res1["notes_filename"]
        faq_file = tdir / res1["faq_filename"]
        notes_text = notes_file.read_text(encoding="utf-8")
        assert "# Meeting Minutes" not in notes_text
        assert "Screenshot Alignment Checks" not in notes_text
        assert "attachment/my%20shot_01.jpg" in notes_text
        assert (tdir / "attachment" / "my shot_01.jpg").is_file()
        assert "Friday." in faq_file.read_text(encoding="utf-8")

        personal_file = tdir / "personal.md"
        personal_file.write_text("do not touch", encoding="utf-8")

        # 2. Collision with overwrite=False refuses and skips LLM FAQ distillation when faq_markdown provided
        calls_before = len(llm.calls)
        conflict_res = json.loads(
            mcp_server.export_to_obsidian(
                vault_path=str(vault),
                minutes_file="team sync.minutes.md",
                chat_history=[{"role": "user", "content": "Ignored"}, {"role": "assistant", "content": "Ignored"}],
                faq_markdown="## Manual FAQ",
                overwrite=False,
            )
        )
        assert len(llm.calls) == calls_before
        assert conflict_res["status"] == "conflict"
        assert conflict_res["written"] is False
        assert conflict_res["exported"] is False
        assert conflict_res["exists"] is True
        assert "already exists" in conflict_res["error"]
        assert "Friday." in faq_file.read_text(encoding="utf-8")
        assert personal_file.read_text(encoding="utf-8") == "do not touch"

        # 3. Confirmed overwrite with faq_markdown="" updates attachments & removes stale FAQ, preserving user file
        res3 = json.loads(
            mcp_server.export_to_obsidian(
                vault_path=str(vault),
                minutes_file="team sync.minutes.md",
                attachment_prefix="new",
                faq_markdown="",
                overwrite=True,
            )
        )
        assert res3["status"] == "exported"
        assert res3["overwritten"] is True
        assert not (tdir / "attachment" / "my shot_01.jpg").exists()
        assert (tdir / "attachment" / "new_01.jpg").is_file()
        assert not faq_file.exists()
        assert personal_file.read_text(encoding="utf-8") == "do not touch"

