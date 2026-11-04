"""Unit tests for ``VideoService.resolve_ffmpeg_binary``.

This guards the regression where chunked transcription failed with
``RuntimeError: ffmpeg is unavailable`` because neither a system ffmpeg nor the
``imageio-ffmpeg`` fallback was available. The fix added ``imageio-ffmpeg`` as a
declared dependency so the fallback always resolves.
"""

from __future__ import annotations

import sys
import types

import pytest

from meeting_notes.video import VideoService


def test_resolve_ffmpeg_prefers_system_binary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "meeting_notes.video.shutil.which", lambda name: "/usr/bin/ffmpeg"
    )
    assert VideoService.resolve_ffmpeg_binary() == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_falls_back_to_imageio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("meeting_notes.video.shutil.which", lambda name: None)

    fake_module = types.ModuleType("imageio_ffmpeg")
    fake_module.get_ffmpeg_exe = lambda: "/fake/imageio/ffmpeg.exe"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_module)

    assert VideoService.resolve_ffmpeg_binary() == "/fake/imageio/ffmpeg.exe"


def test_resolve_ffmpeg_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("meeting_notes.video.shutil.which", lambda name: None)

    def _missing_import(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ImportError("No module named 'imageio_ffmpeg'")
        return __import__(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
    monkeypatch.setattr("builtins.__import__", _missing_import)

    with pytest.raises(RuntimeError, match="ffmpeg is unavailable"):
        VideoService.resolve_ffmpeg_binary()


def test_resolve_ffmpeg_real_environment():
    """The declared ``imageio-ffmpeg`` dependency must resolve a real binary."""
    assert VideoService.resolve_ffmpeg_binary()


def test_plan_transcript_chunk_windows_with_overlap():
    windows = VideoService._plan_transcript_chunk_windows(
        duration_seconds=200,
        chunk_seconds=90,
        overlap_seconds=10,
    )

    assert windows == [
        {"start_seconds": 0.0, "end_seconds": 90.0},
        {"start_seconds": 80.0, "end_seconds": 170.0},
        {"start_seconds": 160.0, "end_seconds": 200.0},
    ]


def test_plan_transcript_chunk_windows_short_duration():
    windows = VideoService._plan_transcript_chunk_windows(
        duration_seconds=45,
        chunk_seconds=90,
        overlap_seconds=10,
    )

    assert windows == [{"start_seconds": 0.0, "end_seconds": 45.0}]


def test_plan_transcript_chunk_windows_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap_seconds must be smaller"):
        VideoService._plan_transcript_chunk_windows(
            duration_seconds=180,
            chunk_seconds=90,
            overlap_seconds=90,
        )


def test_trim_overlapping_prefix_removes_boundary_duplicate():
    previous = "We decided to launch on Friday and Alice will send notes."
    current = "launch on Friday and Alice will send notes. Bob will update Jira."

    trimmed = VideoService._trim_overlapping_prefix(previous, current)

    assert trimmed == "Bob will update Jira."


def test_trim_overlapping_prefix_leaves_non_overlapping_text():
    previous = "Budget review and quarterly targets were discussed."
    current = "Different topic: platform migration risks."

    trimmed = VideoService._trim_overlapping_prefix(previous, current)

    assert trimmed == current
