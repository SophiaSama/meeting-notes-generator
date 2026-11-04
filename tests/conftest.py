"""Shared fixtures for the meeting-notes-helper test suite.

The end-to-end tests exercise the *real* local pipeline (video decode, notes
assembly, markdown/HTML rendering, artifact writing). Only the two external
boundaries that require the network or ffmpeg are faked:

* ``LLMService.call_llm`` (chat completion -> meeting notes)
* ``VideoService.transcribe_video_chunked`` (ffmpeg split + audio transcription)

The fake transcription still writes genuine transcript artifacts to disk so the
pipeline's downstream file references remain real.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from meeting_notes.llm import LLMService
from meeting_notes.notes import NotesService
from meeting_notes.pipeline import PipelineService
from meeting_notes.video import VideoService

# Canned notes returned by the fake LLM. Contains exactly two screenshot
# placeholders to match the two frames produced by ``synthetic_video``.
CANNED_NOTES = """## Meeting Metadata
Recording reviewed by the automated test suite.

## Executive Summary
The team agreed to ship the meeting-notes pipeline.
@@SCREENSHOT_1@@

## Decisions Made
- Adopt the LLM provider for all model calls.

## Action Items
- **Alice**: finalize the export format.
@@SCREENSHOT_2@@

## Next Steps
Reconvene next week.
"""


class FakeLLMService(LLMService):
    """LLM stub that records calls and returns canned meeting notes."""

    def __init__(self, reply: str = CANNED_NOTES):
        self.reply = reply
        self.calls: list[dict] = []

    def call_llm(self, messages, **kwargs):  # type: ignore[override]
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.reply


def _write_fake_transcript(video_path: str) -> dict:
    """Write genuine transcript artifacts and return the pipeline's tx dict."""
    video = Path(video_path)
    transcripts_dir = Path.cwd() / "artifacts" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = transcripts_dir / f"{video.stem}.transcript.txt"
    transcript_plain_path = transcripts_dir / f"{video.stem}.full.txt"
    manifest_path = transcripts_dir / f"{video.stem}.manifest.json"

    transcript_text = "[Chunk 1]\nHello team, welcome to the test meeting.\n"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    transcript_plain_path.write_text(transcript_text, encoding="utf-8")
    manifest = {"coverage_ratio": 1.0, "status": "ok", "actual_chunks": 1}
    manifest_path.write_text("{}", encoding="utf-8")

    return {
        "transcript_text": transcript_text,
        "transcript_path": str(transcript_path),
        "transcript_plain_path": str(transcript_plain_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


class FakeVideoService(VideoService):
    """Real video decode, but fake the ffmpeg+network transcription step."""

    def transcribe_video_chunked(self, video_path, **kwargs):  # type: ignore[override]
        return _write_fake_transcript(video_path)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a deterministic 6-second, 10 fps MP4 (yields exactly 2 frames
    at a 2s interval) so the pipeline can run real frame extraction."""
    video_path = tmp_path / "team_sync.mp4"
    width, height, fps, total_frames = 320, 240, 10, 60

    for fourcc_code in ("mp4v", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        target = video_path if fourcc_code == "mp4v" else video_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            continue
        for i in range(total_frames):
            frame = np.full((height, width, 3), (i * 4 % 256), dtype=np.uint8)
            cv2.putText(
                frame, f"frame {i}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            writer.write(frame)
        writer.release()
        if target.exists() and target.stat().st_size > 0:
            return target

    pytest.skip("No working OpenCV VideoWriter codec available on this host.")


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run inside an isolated working directory so the pipeline writes its
    ``artifacts/`` tree into the temp dir instead of the real workspace."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    return work_dir


@pytest.fixture
def pipeline(isolated_cwd: Path) -> PipelineService:
    """A PipelineService wired with real notes/video decode and fake LLM."""
    llm = FakeLLMService()
    video = FakeVideoService(llm)
    notes = NotesService()
    return PipelineService(llm, video, notes)
