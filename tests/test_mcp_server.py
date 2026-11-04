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
