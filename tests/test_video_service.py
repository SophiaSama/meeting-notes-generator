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

from meeting_notes.video import (
    NoAudioStreamError,
    TranscriptionConfigError,
    VideoService,
)


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


def test_stderr_indicates_no_audio_detects_video_only_input():
    stderr = (
        "Output #0, mp3, to 'audio_chunk_001.mp3':\n"
        "[out#0/mp3 @ 0x136105d80] Output file does not contain any stream\n"
        "Error opening output files: Invalid argument\n"
    )
    assert VideoService._stderr_indicates_no_audio(stderr)
    assert not VideoService._stderr_indicates_no_audio("Invalid data found when processing input")
    assert not VideoService._stderr_indicates_no_audio("")


def test_transcribe_video_chunked_raises_clear_error_for_video_without_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Regression: macOS screen recordings without audio crashed with a raw ffmpeg dump."""
    import subprocess

    from meeting_notes.llm import LLMService

    ffmpeg_bin = VideoService.resolve_ffmpeg_binary()
    video_file = tmp_path / "silent.mp4"
    gen = subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt", "yuv420p", str(video_file),
        ],
        capture_output=True,
        text=True,
    )
    if gen.returncode != 0:
        pytest.skip(f"ffmpeg cannot synthesize a test video: {gen.stderr[-200:]}")

    monkeypatch.chdir(tmp_path)
    llm = LLMService()
    vs = VideoService(llm)

    class FakeClient:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    raise AssertionError("transcription must not be attempted without audio")

    monkeypatch.setattr(llm, "build_client", lambda base_url, api_key: FakeClient())

    with pytest.raises(NoAudioStreamError, match="No audio detected"):
        vs.transcribe_video_chunked(
            str(video_file),
            base_url="http://localhost:4000",
            api_key="k",
            transcribe_model="gpt-4o-transcribe",
        )


def test_is_gemini_transcribe_model():
    assert VideoService.is_gemini_transcribe_model("gemini-3.5-transcribe")
    assert VideoService.is_gemini_transcribe_model("models/gemini-3.5-transcribe")
    assert VideoService.is_gemini_transcribe_model("GEMINI-3.5-TRANSCRIBE")
    assert VideoService.is_gemini_transcribe_model("  models/gemini-3.5-transcribe  ")
    assert VideoService.is_gemini_transcribe_model("gemini-transcribe")
    assert not VideoService.is_gemini_transcribe_model("gpt-4o-transcribe")
    assert not VideoService.is_gemini_transcribe_model("whisper-large-v3-turbo")
    assert not VideoService.is_gemini_transcribe_model("whisper-1")
    assert not VideoService.is_gemini_transcribe_model("")
    assert not VideoService.is_gemini_transcribe_model(None)


def test_validate_transcription_config_rejects_non_gemini_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LLM_TRANSCRIBE_BASE_URL", raising=False)
    with pytest.raises(TranscriptionConfigError) as exc_info:
        VideoService.validate_transcription_config("gpt-4o-transcribe", base_url="")
    msg = str(exc_info.value)
    assert "LLM_TRANSCRIBE_BASE_URL" in msg
    assert "gemini-3.5-transcribe" in msg
    assert "gpt-4o-transcribe" in msg


def test_validate_transcription_config_accepts_non_gemini_with_base_url(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LLM_TRANSCRIBE_BASE_URL", raising=False)
    # Passed explicitly via base_url argument
    VideoService.validate_transcription_config(
        "gpt-4o-transcribe", base_url="https://api.openai.com/v1"
    )
    # Or set in env
    monkeypatch.setenv("LLM_TRANSCRIBE_BASE_URL", "https://api.groq.com/openai/v1")
    VideoService.validate_transcription_config("whisper-large-v3-turbo", base_url="")


def test_validate_transcription_config_accepts_gemini_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LLM_TRANSCRIBE_BASE_URL", raising=False)
    VideoService.validate_transcription_config("gemini-3.5-transcribe", base_url="")
    VideoService.validate_transcription_config(
        "models/gemini-3.5-transcribe", base_url=""
    )


def test_transcribe_video_chunked_fails_fast_on_misconfigured_non_gemini(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_TRANSCRIBE_BASE_URL", raising=False)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    with pytest.raises(TranscriptionConfigError, match="LLM_TRANSCRIBE_BASE_URL"):
        vs.transcribe_video_chunked(
            str(video_file),
            base_url="",
            api_key="",
            transcribe_model="gpt-4o-transcribe",
        )


def test_transcribe_video_chunked_non_gemini_with_base_url_uses_openai_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    llm = LLMService()
    vs = VideoService(llm)

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    called_models = []

    class FakeClient:
        class audio:
            class transcriptions:
                @staticmethod
                def create(*, model, file):
                    called_models.append(model)
                    return "OpenAI transcribe response"

    monkeypatch.setattr(llm, "build_client", lambda base_url, api_key: FakeClient())

    tx = vs.transcribe_video_chunked(
        str(video_file),
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        transcribe_model="whisper-large-v3-turbo",
        chunk_seconds=90,
    )

    assert called_models == ["whisper-large-v3-turbo"]
    assert "OpenAI transcribe response" in tx["transcript_text"]
    assert tx["manifest"]["model"] == "whisper-large-v3-turbo"


def test_transcribe_video_chunked_gemini_uses_google_genai_path_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_TRANSCRIBE_BASE_URL", raising=False)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 190.0,
            "duration_str": "3:10",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 1.0,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    uploaded_files = []
    created_interactions = []
    deleted_file_names = []

    class FakeUploadedFile:
        def __init__(self, name: str):
            self.name = name

    class FakeInteraction:
        def __init__(self, output_text: str):
            self.output_text = output_text

    class FakeFilesAPI:
        def upload(self, *, file: str):
            f = FakeUploadedFile(f"files/{Path(file).stem}")
            uploaded_files.append(f)
            return f

        def delete(self, *, name: str):
            deleted_file_names.append(name)

    chunk_replies = {
        "files/audio_chunk_001": "Gemini first chunk with overlap phrase alpha beta gamma delta epsilon.",
        "files/audio_chunk_002": "overlap phrase alpha beta gamma delta epsilon. Gemini second chunk unique.",
        "files/audio_chunk_003": "Gemini third chunk ending.",
    }

    class FakeInteractionsAPI:
        def create(self, *, model, input, transcription_config=None, **kwargs):
            created_interactions.append(
                {
                    "model": model,
                    "input": input,
                    "transcription_config": transcription_config,
                    "kwargs": kwargs,
                }
            )
            file_obj = input[0]
            return FakeInteraction(chunk_replies[file_obj.name])

    class FakeGeminiClient:
        def __init__(self):
            self.files = FakeFilesAPI()
            self.interactions = FakeInteractionsAPI()

    fake_client = FakeGeminiClient()
    monkeypatch.setattr(vs, "build_gemini_client", lambda api_key=None: fake_client)

    tx = vs.transcribe_video_chunked(
        str(video_file),
        base_url="",
        api_key="gemini-key",
        transcribe_model="gemini-3.5-transcribe",
        chunk_seconds=90,
        chunk_overlap_seconds=10,
    )

    assert len(uploaded_files) == 3
    assert len(created_interactions) == 3
    assert len(deleted_file_names) == 3
    assert set(deleted_file_names) == {f.name for f in uploaded_files}
    for interaction_call in created_interactions:
        assert interaction_call["model"] == "gemini-3.5-transcribe"
        assert interaction_call["transcription_config"] == {"mode": "smart"}
        assert interaction_call["kwargs"]["generation_config"] == {
            "transcription_config": {"mode": "smart"}
        }
    assert tx["manifest"]["actual_chunks"] == 3
    assert "[Chunk 1]\nGemini first chunk" in tx["transcript_text"]
    assert "[Chunk 2]\nGemini second chunk unique." in tx["transcript_text"]
    assert "[Chunk 3]\nGemini third chunk ending." in tx["transcript_text"]


def test_transcribe_video_chunked_gemini_cleans_up_file_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    deleted_names = []

    class FakeUploadedFile:
        name = "files/test-chunk-fail"

    class FailingInteractionsAPI:
        def create(self, **kwargs):
            raise RuntimeError("Gemini API call failed")

    class FakeFilesAPI:
        def upload(self, *, file: str):
            return FakeUploadedFile()

        def delete(self, *, name: str):
            deleted_names.append(name)

    class FakeGeminiClient:
        files = FakeFilesAPI()
        interactions = FailingInteractionsAPI()

    monkeypatch.setattr(
        vs, "build_gemini_client", lambda api_key=None: FakeGeminiClient()
    )

    with pytest.raises(RuntimeError, match="Gemini API call failed"):
        vs.transcribe_video_chunked(
            str(video_file),
            transcribe_model="models/gemini-3.5-transcribe",
        )

    # Remote file deletion must have been invoked in finally block despite the exception
    assert deleted_names == ["files/test-chunk-fail"]


def test_validate_transcription_config_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_TRANSCRIBE_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_TRANSCRIBE_MODEL", raising=False)

    # Defaults to gemini-3.5-transcribe when model is None or empty, so no error without base_url
    VideoService.validate_transcription_config(None)
    VideoService.validate_transcription_config("")
    VideoService.validate_transcription_config()

    # When LLM_TRANSCRIBE_MODEL is set in env to a non-Gemini model and model is None
    monkeypatch.setenv("LLM_TRANSCRIBE_MODEL", "whisper-large-v3-turbo")
    with pytest.raises(TranscriptionConfigError, match="LLM_TRANSCRIBE_BASE_URL"):
        VideoService.validate_transcription_config(None)


def test_build_gemini_client_key_precedence(monkeypatch: pytest.MonkeyPatch):
    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())
    constructed_keys = []

    class FakeGenAIClient:
        def __init__(self, api_key=None):
            constructed_keys.append(api_key)

    monkeypatch.setattr("google.genai.Client", FakeGenAIClient)

    # 1. Explicit api_key overrides env
    monkeypatch.setenv("LLM_TRANSCRIBE_API_KEY", "env-transcribe-key")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
    vs.build_gemini_client(api_key="explicit-key")
    assert constructed_keys[-1] == "explicit-key"

    # 2. LLM_TRANSCRIBE_API_KEY takes precedence when api_key is None/empty
    vs.build_gemini_client()
    assert constructed_keys[-1] == "env-transcribe-key"

    # 3. GEMINI_API_KEY
    monkeypatch.delenv("LLM_TRANSCRIBE_API_KEY", raising=False)
    vs.build_gemini_client("")
    assert constructed_keys[-1] == "env-gemini-key"

    # 4. GOOGLE_API_KEY
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    vs.build_gemini_client(None)
    assert constructed_keys[-1] == "env-google-key"

    # 5. No key at all -> calls genai.Client() without api_key kwarg
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    vs.build_gemini_client()
    assert constructed_keys[-1] is None


def test_transcribe_video_chunked_gemini_prefers_gemini_env_key_over_chat_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    captured_api_key = []

    class FakeInteraction:
        output_text = "Transcribed text"

    class FakeClient:
        class files:
            @staticmethod
            def upload(*, file):
                class U:
                    name = "files/test"
                return U()

            @staticmethod
            def delete(*, name):
                pass

        class interactions:
            @staticmethod
            def create(**kwargs):
                return FakeInteraction()

    monkeypatch.setattr(
        vs,
        "build_gemini_client",
        lambda api_key=None: (captured_api_key.append(api_key), FakeClient())[1],
    )

    # Set GEMINI_API_KEY in env, but pass chat OpenAI key in api_key arg
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-correct-gemini-key")
    vs.transcribe_video_chunked(
        str(video_file),
        api_key="sk-openai-chat-key",
        transcribe_model="gemini-3.5-transcribe",
    )

    assert captured_api_key == ["AIzaSy-correct-gemini-key"]


def test_transcribe_video_chunked_gemini_handles_empty_output_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    class FakeInteractionWithNoneOutput:
        output_text = None

        def __str__(self):
            return "<Interaction id='should-not-appear-in-transcript'>"

    class FakeClient:
        class files:
            @staticmethod
            def upload(*, file):
                class U:
                    name = "files/test"
                return U()

            @staticmethod
            def delete(*, name):
                pass

        class interactions:
            @staticmethod
            def create(**kwargs):
                return FakeInteractionWithNoneOutput()

    monkeypatch.setattr(vs, "build_gemini_client", lambda api_key=None: FakeClient())

    tx = vs.transcribe_video_chunked(
        str(video_file),
        transcribe_model="gemini-3.5-transcribe",
    )

    assert "should-not-appear-in-transcript" not in tx["transcript_text"]
    assert "[Chunk 1]" in tx["transcript_text"]


def test_build_gemini_client_wraps_interactions_create_for_real_sdk():
    """Verify build_gemini_client returns a client whose create method adapts transcription_config."""
    from meeting_notes.llm import LLMService
    from unittest.mock import MagicMock
    import httpx

    vs = VideoService(LLMService())
    client = vs.build_gemini_client(api_key="fake-key")

    fake_http_res = httpx.Response(
        200,
        json={
            "id": "inter-test",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Real SDK wrapped transcript"}],
                }
            ],
        },
        request=httpx.Request("POST", "https://fake"),
    )
    client.interactions.do_request = MagicMock(return_value=fake_http_res)

    res = client.interactions.create(
        model="gemini-3.5-transcribe",
        input="test audio input",
        transcription_config={"mode": "smart"},
    )
    assert res.output_text == "Real SDK wrapped transcript"
    req_body = client.interactions.do_request.call_args.kwargs["request"].content.decode()
    assert '"transcription_config":{"mode":"smart"}' in req_body


def test_transcribe_video_chunked_gemini_with_real_sdk_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Test full transcribe_video_chunked execution with real google.genai.Client instance."""
    from unittest.mock import MagicMock
    import httpx
    from google.genai import types

    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    client = vs.build_gemini_client(api_key="fake-test-key")

    real_file = types.File(
        name="files/test-real-chunk",
        uri="https://generativelanguage.googleapis.com/v1beta/files/test-real-chunk",
        mime_type="audio/mp3",
    )
    deleted_names = []
    client.files.upload = MagicMock(return_value=real_file)
    client.files.delete = lambda *, name: deleted_names.append(name)

    fake_http_res = httpx.Response(
        200,
        json={
            "id": "inter-real-chunk",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Real SDK transcribed content."}],
                }
            ],
        },
        request=httpx.Request("POST", "https://fake"),
    )
    client.interactions.do_request = MagicMock(return_value=fake_http_res)

    monkeypatch.setattr(vs, "build_gemini_client", lambda api_key=None: client)

    tx = vs.transcribe_video_chunked(
        str(video_file),
        transcribe_model="gemini-3.5-transcribe",
    )

    assert deleted_names == ["files/test-real-chunk"]
    assert "Real SDK transcribed content." in tx["transcript_text"]
    assert tx["manifest"]["actual_chunks"] == 1


def test_transcribe_video_chunked_gemini_raw_unwrapped_client_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Test transcribe_video_chunked handles raw unwrapped genai.Client instances."""
    from unittest.mock import MagicMock
    import httpx
    from google import genai
    from google.genai import types

    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    raw_client = genai.Client(api_key="fake-raw-key")
    real_file = types.File(
        name="files/test-raw-chunk",
        uri="https://generativelanguage.googleapis.com/v1beta/files/test-raw-chunk",
        mime_type="audio/mp3",
    )
    deleted_names = []
    raw_client.files.upload = MagicMock(return_value=real_file)
    raw_client.files.delete = lambda *, name: deleted_names.append(name)

    fake_http_res = httpx.Response(
        200,
        json={
            "id": "inter-raw-chunk",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Raw client transcribed content."}],
                }
            ],
        },
        request=httpx.Request("POST", "https://fake"),
    )
    raw_client.interactions.do_request = MagicMock(return_value=fake_http_res)

    monkeypatch.setattr(vs, "build_gemini_client", lambda api_key=None: raw_client)

    tx = vs.transcribe_video_chunked(
        str(video_file),
        transcribe_model="gemini-3.5-transcribe",
    )

    assert deleted_names == ["files/test-raw-chunk"]
    assert "Raw client transcribed content." in tx["transcript_text"]


def test_transcribe_video_chunked_non_gemini_resolves_env_transcribe_api_key_when_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """When api_key is empty, non-Gemini transcribe must resolve LLM_TRANSCRIBE_API_KEY from env."""
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    llm = LLMService()
    vs = VideoService(llm)

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    captured_client_args = []

    class FakeClient:
        class audio:
            class transcriptions:
                @staticmethod
                def create(*, model, file):
                    return "Non-gemini env key transcript"

    monkeypatch.setattr(
        llm,
        "build_client",
        lambda base_url, api_key: (captured_client_args.append((base_url, api_key)), FakeClient())[1],
    )
    monkeypatch.setenv("LLM_TRANSCRIBE_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_TRANSCRIBE_API_KEY", "gsk-dedicated-env-transcribe-key")

    tx = vs.transcribe_video_chunked(
        str(video_file),
        base_url="",
        api_key="",
        transcribe_model="whisper-large-v3-turbo",
        chunk_seconds=90,
    )

    assert captured_client_args == [
        ("https://api.groq.com/openai/v1", "gsk-dedicated-env-transcribe-key")
    ]
    assert "Non-gemini env key transcript" in tx["transcript_text"]


def test_transcribe_video_chunked_gemini_strict_transcription_config_mock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Mocks that strictly reject generation_config must gracefully fallback to transcription_config."""
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    class StrictMockInteractions:
        def create(self, *, model, input, transcription_config=None):
            class Resp:
                text = "Strict mock transcribed text"
            return Resp()

    deleted_names = []

    class StrictClient:
        class files:
            @staticmethod
            def upload(*, file):
                return {"name": "files/strict-chunk-1"}

            @staticmethod
            def delete(*, name):
                deleted_names.append(name)

        interactions = StrictMockInteractions()

    monkeypatch.setattr(vs, "build_gemini_client", lambda api_key=None: StrictClient())

    tx = vs.transcribe_video_chunked(
        str(video_file),
        transcribe_model="gemini-3.5-transcribe",
    )

    assert deleted_names == ["files/strict-chunk-1"]
    assert "Strict mock transcribed text" in tx["transcript_text"]


def test_transcribe_video_chunked_gemini_recovers_from_input_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """When raw client raises ValueError/Pydantic validation on bare File input, fallback recovers with audio dict."""
    monkeypatch.chdir(tmp_path)
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"fake-video")

    from meeting_notes.llm import LLMService

    vs = VideoService(LLMService())

    monkeypatch.setattr(
        vs,
        "get_video_info",
        lambda path: {
            "duration": 50.0,
            "duration_str": "0:50",
            "fps": 25,
            "resolution": "1280x720",
            "size_mb": 0.5,
            "filename": "sample.mp4",
        },
    )
    monkeypatch.setattr(vs, "resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd, capture_output=False, text=False):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake-mp3-bytes")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("meeting_notes.video.subprocess.run", fake_run)

    class CustomFile:
        name = "files/val-chunk"
        uri = "https://generativelanguage.googleapis.com/v1beta/files/val-chunk"
        mime_type = "audio/mp3"

    recorded_inputs = []

    class MockInteractionsWithValidation:
        def create(self, *, model, input, **kwargs):
            recorded_inputs.append(input)
            if any(isinstance(x, CustomFile) for x in input):
                raise ValueError("Input should be a valid dictionary or instance of AudioContent")
            class Resp:
                output_text = "Recovered from validation error text"
            return Resp()

    deleted_names = []

    class MockClient:
        class files:
            @staticmethod
            def upload(*, file):
                return CustomFile()

            @staticmethod
            def delete(*, name):
                deleted_names.append(name)

        interactions = MockInteractionsWithValidation()

    monkeypatch.setattr(vs, "build_gemini_client", lambda api_key=None: MockClient())

    tx = vs.transcribe_video_chunked(
        str(video_file),
        transcribe_model="gemini-3.5-transcribe",
    )

    assert deleted_names == ["files/val-chunk"]
    assert "Recovered from validation error text" in tx["transcript_text"]
    assert len(recorded_inputs) == 2
    assert isinstance(recorded_inputs[0][0], CustomFile)
    assert isinstance(recorded_inputs[1][0], dict)
    assert recorded_inputs[1][0]["type"] == "audio"
    assert recorded_inputs[1][0]["uri"] == "https://generativelanguage.googleapis.com/v1beta/files/val-chunk"


