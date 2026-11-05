"""Unit tests for ``VideoService.resolve_ffmpeg_binary``.

This guards the regression where chunked transcription failed with
``RuntimeError: ffmpeg is unavailable`` because neither a system ffmpeg nor the
``imageio-ffmpeg`` fallback was available. The fix added ``imageio-ffmpeg`` as a
declared dependency so the fallback always resolves.
"""

from __future__ import annotations

from pathlib import Path
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


def test_transcribe_video_chunked_runs_chunks_in_parallel_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import threading
    from meeting_notes.llm import LLMService

    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    llm = LLMService()
    vs = VideoService(llm)

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 200.0,
            "duration_str": "3:20",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 1.0,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    barrier = threading.Barrier(3, timeout=3.0)

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")
        barrier.wait()

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    chunk_texts = {
        "audio_chunk_001.mp3": "First chunk content with overlap words alpha beta gamma delta epsilon.",
        "audio_chunk_002.mp3": "overlap words alpha beta gamma delta epsilon. Second chunk unique text.",
        "audio_chunk_003.mp3": "Third chunk conclusion.",
    }

    class FakeTranscriptions:
        def create(self, *, model, file):
            name = Path(file.name).name
            return chunk_texts[name]

    class FakeClient:
        class audio:
            transcriptions = FakeTranscriptions()

    monkeypatch.setattr(llm, "build_client", lambda base_url, api_key: FakeClient())

    tx = vs.transcribe_video_chunked(
        str(video_file),
        base_url="http://localhost:4000",
        api_key="k",
        transcribe_model="gpt-4o-transcribe",
        chunk_seconds=90,
        chunk_overlap_seconds=10,
    )

    assert tx["manifest"]["actual_chunks"] == 3
    assert "[Chunk 1]\nFirst chunk content" in tx["transcript_text"]
    assert "[Chunk 2]\nSecond chunk unique text." in tx["transcript_text"]
    assert "[Chunk 3]\nThird chunk conclusion." in tx["transcript_text"]
    assert [w["chunk_index"] for w in tx["manifest"]["chunk_windows"]] == [1, 2, 3]
