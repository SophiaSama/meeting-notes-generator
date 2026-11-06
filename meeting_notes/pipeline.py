import json
import logging
import math
import os
import re
import shutil
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from html import escape
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from .llm import LLMOutputLimitError, LLMService
from .notes import NotesService
from .video import NoAudioStreamError, TranscriptionConfigError, VideoService

logger = logging.getLogger(__name__)


class MeetingContext(dict):
    """Serializable session context for follow-up meeting Q&A."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class PipelineRunResult(tuple):
    """4-tuple (thumbnails, notes_text, html_notes, out_path) that also carries .meeting_context."""

    meeting_context: MeetingContext | None

    def __new__(cls, items, meeting_context: MeetingContext | None = None):
        instance = super().__new__(cls, items)
        instance.meeting_context = meeting_context
        return instance

    @property
    def context(self) -> MeetingContext | None:
        return self.meeting_context


class PipelineService:
    """End-to-end generation pipeline backing the Gradio actions."""

    DEFAULT_CHAT_REQUEST_LIMIT_BYTES = 10_485_760
    CHAT_REQUEST_SAFETY_RATIO = 0.90
    CHAT_REQUEST_TEXT_RESERVE_BYTES = 1_000_000
    DEFAULT_SCREENSHOT_REQUEST_BYTES = 250_000
    DEFAULT_DIRECT_TRANSCRIPT_CHAR_LIMIT = 30_000
    DEFAULT_MAP_REDUCE_TRIGGER_CHARS = 28_000
    DEFAULT_MAP_REDUCE_SEGMENT_CHARS = 12_000
    DEFAULT_USE_BATCH_MAP_REDUCE = False
    DEFAULT_SEGMENT_MAX_TOKENS = 4_000
    DEFAULT_REDUCE_MAX_TOKENS = 3_500
    DEFAULT_FINAL_MAX_TOKENS = 8_000
    DEFAULT_QA_MAX_TOKENS = 4_000
    DEFAULT_LENGTH_RETRY_ENABLED = True
    DEFAULT_LENGTH_RETRY_TOKEN_MULTIPLIER = 1.5
    # Reasoning ("thinking") models spend part of max_tokens on hidden
    # reasoning, so they need much larger budgets than non-thinking models.
    DEFAULT_REASONING_SEGMENT_MAX_TOKENS = 16_000
    DEFAULT_REASONING_REDUCE_MAX_TOKENS = 16_000
    DEFAULT_REASONING_FINAL_MAX_TOKENS = 24_000
    DEFAULT_REASONING_QA_MAX_TOKENS = 16_000
    DEFAULT_LENGTH_RETRY_MAX_TOKENS = 32_000
    DEFAULT_FALLBACK_MAX_TOKENS = 16_000
    DEFAULT_OCR_MAX_WORKERS = 4

    @classmethod
    def _ocr_max_workers(cls) -> int:
        raw = os.getenv("LLM_OCR_MAX_WORKERS", "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return cls.DEFAULT_OCR_MAX_WORKERS

    def __init__(
        self,
        llm_service: LLMService,
        video_service: VideoService,
        notes_service: NotesService,
    ):
        self.llm_service = llm_service
        self.video_service = video_service
        self.notes_service = notes_service

    @staticmethod
    def _detect_local_timezone() -> str:
        """Return a human-readable local timezone label, e.g. 'SGT (UTC+08:00)'.

        Used to give the LLM accurate timezone context so it doesn't have to
        guess from participant names or meeting content.
        """
        now = datetime.now(timezone.utc).astimezone()
        utc_offset = now.strftime("%z")          # e.g. "+0800"
        sign = utc_offset[0]
        hours = utc_offset[1:3]
        minutes = utc_offset[3:5]
        utc_label = f"UTC{sign}{hours}:{minutes}"

        # Try to get the IANA timezone name (e.g. "Asia/Singapore")
        try:
            import zoneinfo
            tz_name = str(now.tzinfo)
        except Exception:
            tz_name = ""

        # Map well-known UTC offsets to common abbreviations
        _offset_abbrevs = {
            "+01:00": "CET", "+02:00": "CEST", "+08:00": "SGT",
            "+09:00": "JST", "+05:30": "IST", "-05:00": "EST",
            "-04:00": "EDT", "-06:00": "CST", "-07:00": "MST",
            "-08:00": "PST", "+00:00": "UTC", "+10:00": "AEST",
        }
        offset_key = f"{sign}{hours}:{minutes}"
        abbrev = _offset_abbrevs.get(offset_key, "")

        if abbrev:
            return f"{abbrev} ({utc_label})"
        if tz_name and tz_name != str(now.tzinfo.__class__):
            return f"{tz_name} ({utc_label})"
        return utc_label

    @staticmethod
    def _resolve_input_video_path(video_file) -> Path:
        """Normalize Gradio video payloads to a filesystem path."""
        if isinstance(video_file, str):
            return Path(video_file)

        if isinstance(video_file, dict):
            for key in ("path", "video", "name"):
                value = video_file.get(key)
                if value:
                    return Path(value)

        named_value = getattr(video_file, "name", None)
        if named_value:
            return Path(named_value)

        raise gr.Error("Unsupported video input format from Gradio.")

    @staticmethod
    def _resolve_input_text_path(text_file) -> Path:
        """Normalize Gradio text file payloads to a filesystem path."""
        if isinstance(text_file, str):
            return Path(text_file)

        if isinstance(text_file, dict):
            for key in ("path", "name"):
                value = text_file.get(key)
                if value:
                    return Path(value)

        named_value = getattr(text_file, "name", None)
        if named_value:
            return Path(named_value)

        raise gr.Error("Unsupported transcript input format from Gradio.")

    @staticmethod
    def _load_transcript_text(transcript_file) -> tuple[Path, str]:
        """Read uploaded transcript text from disk with a safe fallback decoder."""
        transcript_path = PipelineService._resolve_input_text_path(transcript_file)
        if not transcript_path.exists():
            raise gr.Error(f"Uploaded transcript path does not exist: {transcript_path}")

        try:
            transcript_text = transcript_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise gr.Error(f"Cannot read transcript file: {transcript_path}") from e

        if not transcript_text.strip():
            raise gr.Error("Uploaded transcript is empty.")

        return transcript_path, transcript_text

    @staticmethod
    def _upload_index_path() -> Path:
        return Path.cwd() / "artifacts" / "uploads" / ".upload_index.json"

    @classmethod
    def _load_upload_index(cls) -> dict:
        index_path = cls._upload_index_path()
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
            if not isinstance(data, dict):
                return {}
            return cls._normalize_index(data)
        # No index yet: backfill from previously staged uploads so re-uploading
        # an older file is detected immediately (and the index file is created).
        return cls._backfill_upload_index()

    @staticmethod
    def _normalize_index(data: dict) -> dict:
        """Key the index by upload filename.

        Older indexes were keyed by content hash; re-key them to the upload's
        ``original_name`` so detection works by filename and legacy entries keep
        working without a rebuild.
        """
        normalized: dict = {}
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            name = entry.get("original_name") or key
            normalized[name] = entry
        return normalized

    @classmethod
    def _backfill_upload_index(cls) -> dict:
        """Build the index from videos already present in the uploads dir."""
        uploads_dir = cls._upload_index_path().parent
        index: dict = {}
        if not uploads_dir.exists():
            return index

        for path in sorted(uploads_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            index.setdefault(
                path.name,
                {
                    "original_name": path.name,
                    "staged_name": path.name,
                    "first_seen": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(),
                },
            )

        cls._save_upload_index(index)
        return index

    @classmethod
    def _save_upload_index(cls, index: dict) -> None:
        index_path = cls._upload_index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _discover_minutes_name(staged_name: str, existing_minutes_name: str | None) -> str | None:
        """Resolve a minutes filename for duplicate notices.

        Legacy upload-index rows may not have ``minutes_name`` populated.
        Fall back to looking for a minutes file derived from the staged video
        stem so the UI can still print an actionable path.
        """
        if existing_minutes_name:
            return existing_minutes_name
        if not staged_name:
            return None

        minutes_dir = Path.cwd() / "artifacts" / "meeting_minutes"
        if not minutes_dir.exists():
            return None

        staged_stem = Path(staged_name).stem
        direct_match = minutes_dir / f"{staged_stem}.minutes.md"
        if direct_match.exists():
            return direct_match.name

        matches = sorted(
            minutes_dir.glob(f"{staged_stem}*.minutes.md"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        return matches[0].name if matches else None

    @classmethod
    def lookup_duplicate(cls, source_path: Path) -> dict | None:
        """Return duplicate metadata if a file with this name was already staged.

        Matching is by upload filename (not content), so re-uploading the same
        recording is detected even when Gradio re-encodes the bytes. Read-only:
        does not copy or modify the index.
        """
        prior = cls._load_upload_index().get(source_path.name)
        if not prior:
            return None

        uploads_dir = cls._upload_index_path().parent
        staged_name = prior.get("staged_name", "")
        prior_path = uploads_dir / staged_name
        if not prior_path.exists():
            return None
        minutes_name = cls._discover_minutes_name(staged_name, prior.get("minutes_name"))
        return {
            "original_name": prior.get("original_name", source_path.name),
            "staged_name": prior.get("staged_name", prior_path.name),
            "first_seen": prior.get("first_seen", ""),
            "minutes_name": minutes_name,
        }

    def check_upload_for_duplicate(self, video_file):
        """Upload-time notice: inform the user when this content was seen before.

        The upload is kept (the user can still generate notes); we surface both a
        warning toast and a persistent banner. Generation later reuses the
        existing staged copy instead of writing a second file to
        ``artifacts/uploads/``.

        Returns a ``gr.update`` for the duplicate banner: shown with a message
        when the content is a known duplicate, hidden otherwise.
        """
        hidden = gr.update(value="", visible=False)
        if video_file is None:
            return hidden
        try:
            source_path = self._resolve_input_video_path(video_file)
        except Exception:
            return hidden

        duplicate = self.lookup_duplicate(source_path)
        if not duplicate:
            return hidden

        first_seen = duplicate.get("first_seen", "")
        when = f" on {first_seen}" if first_seen else ""
        minutes_name = duplicate.get("minutes_name")
        minutes_hint = (
            f" Existing notes: artifacts/meeting_minutes/{minutes_name}."
            if minutes_name
            else ""
        )
        message = (
            f"Duplicate video detected: this file was already uploaded{when}, so "
            f"it will not be copied to artifacts again.{minutes_hint} You can "
            "still generate notes from it."
        )
        try:
            gr.Warning(message)
        except Exception:
            pass
        safe_message = escape(message)
        banner_html = (
            "<div style='padding:10px 12px;border:1px solid #f0c36d;"
            "background:#fff8e1;border-radius:6px;color:#7a5c00;'>"
            "&#9888;&#65039; <b>Duplicate video detected.</b> "
            f"{safe_message}"
            "</div>"
        )
        return gr.update(value=banner_html, visible=True)

    @classmethod
    def _stage_video_copy(cls, source_path: Path) -> tuple[Path, dict | None]:
        """Copy upload out of Gradio temp directory to a stable local path.

        Detects content duplicates (same bytes, regardless of filename) using a
        SHA-256 index. When a previously staged copy still exists on disk it is
        reused instead of writing another copy, and duplicate metadata is
        returned so the caller can notify the user.
        """
        uploads_dir = Path.cwd() / "artifacts" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        index = cls._load_upload_index()

        prior = index.get(source_path.name)
        if prior:
            prior_path = uploads_dir / prior.get("staged_name", "")
            if prior_path.exists():
                return prior_path, {
                    "original_name": prior.get("original_name", source_path.name),
                    "staged_name": prior.get("staged_name", prior_path.name),
                    "first_seen": prior.get("first_seen", ""),
                }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = source_path.suffix or ".webm"
        staged_name = f"{source_path.stem}_{timestamp}{suffix}"
        staged_path = uploads_dir / staged_name

        # Gradio temp files on Windows can be briefly locked right after upload.
        last_error: Exception | None = None
        for _ in range(5):
            try:
                shutil.copy2(source_path, staged_path)
                break
            except PermissionError as e:
                last_error = e
                time.sleep(0.35)
        else:
            if last_error is not None:
                raise last_error

        index[source_path.name] = {
            "original_name": source_path.name,
            "staged_name": staged_name,
            "first_seen": datetime.now().isoformat(),
        }
        cls._save_upload_index(index)
        return staged_path, None

    @classmethod
    def _record_minutes_for_video(cls, staged_path: Path, minutes_name: str) -> None:
        """Attach the generated minutes filename to this video's index entry."""
        if not staged_path.exists():
            return
        index = cls._load_upload_index()
        for entry in index.values():
            if entry.get("staged_name") == staged_path.name:
                entry["minutes_name"] = minutes_name
                cls._save_upload_index(index)
                return

    @classmethod
    def _direct_transcript_char_limit(cls) -> int:
        raw_limit = os.getenv("LLM_DIRECT_TRANSCRIPT_CHAR_LIMIT", "").strip()
        try:
            value = int(raw_limit) if raw_limit else cls.DEFAULT_DIRECT_TRANSCRIPT_CHAR_LIMIT
        except ValueError:
            value = cls.DEFAULT_DIRECT_TRANSCRIPT_CHAR_LIMIT
        return max(1_000, value)

    @classmethod
    def _map_reduce_trigger_chars(cls) -> int:
        raw_limit = os.getenv("LLM_NOTES_MAP_REDUCE_TRIGGER_CHARS", "").strip()
        try:
            value = int(raw_limit) if raw_limit else cls.DEFAULT_MAP_REDUCE_TRIGGER_CHARS
        except ValueError:
            value = cls.DEFAULT_MAP_REDUCE_TRIGGER_CHARS
        return max(100, value)

    @classmethod
    def _map_reduce_segment_chars(cls) -> int:
        raw_limit = os.getenv("LLM_NOTES_MAP_REDUCE_SEGMENT_CHARS", "").strip()
        try:
            value = int(raw_limit) if raw_limit else cls.DEFAULT_MAP_REDUCE_SEGMENT_CHARS
        except ValueError:
            value = cls.DEFAULT_MAP_REDUCE_SEGMENT_CHARS
        return max(500, value)

    @classmethod
    def _use_batch_map_reduce(cls) -> bool:
        raw_value = os.getenv("LLM_NOTES_USE_BATCH_MAP_REDUCE", "").strip().lower()
        if not raw_value:
            return cls.DEFAULT_USE_BATCH_MAP_REDUCE
        return raw_value in {"1", "true", "yes", "on"}

    @staticmethod
    def _positive_int_env(name: str, default: int, minimum: int = 1) -> int:
        raw_value = os.getenv(name, "").strip()
        try:
            value = int(raw_value) if raw_value else default
        except ValueError:
            value = default
        return max(minimum, value)

    @classmethod
    def _stage_max_tokens(
        cls, env_name: str, default: int, reasoning_default: int, model: str | None
    ) -> int:
        """Explicit env var wins; otherwise pick a model-appropriate default."""
        if not os.getenv(env_name, "").strip() and model and LLMService._is_reasoning_model(model):
            default = reasoning_default
        return cls._positive_int_env(env_name, default, 128)

    @classmethod
    def _segment_max_tokens(cls, model: str | None = None) -> int:
        return cls._stage_max_tokens(
            "LLM_NOTES_SEGMENT_MAX_TOKENS",
            cls.DEFAULT_SEGMENT_MAX_TOKENS,
            cls.DEFAULT_REASONING_SEGMENT_MAX_TOKENS,
            model,
        )

    @classmethod
    def _reduce_max_tokens(cls, model: str | None = None) -> int:
        return cls._stage_max_tokens(
            "LLM_NOTES_REDUCE_MAX_TOKENS",
            cls.DEFAULT_REDUCE_MAX_TOKENS,
            cls.DEFAULT_REASONING_REDUCE_MAX_TOKENS,
            model,
        )

    @classmethod
    def _final_max_tokens(cls, model: str | None = None) -> int:
        return cls._stage_max_tokens(
            "LLM_NOTES_FINAL_MAX_TOKENS",
            cls.DEFAULT_FINAL_MAX_TOKENS,
            cls.DEFAULT_REASONING_FINAL_MAX_TOKENS,
            model,
        )

    @classmethod
    def _qa_max_tokens(cls, model: str | None = None) -> int:
        return cls._stage_max_tokens(
            "LLM_NOTES_QA_MAX_TOKENS",
            cls.DEFAULT_QA_MAX_TOKENS,
            cls.DEFAULT_REASONING_QA_MAX_TOKENS,
            model,
        )

    @classmethod
    def _length_retry_max_tokens(cls) -> int:
        return cls._positive_int_env(
            "LLM_NOTES_LENGTH_RETRY_MAX_TOKENS", cls.DEFAULT_LENGTH_RETRY_MAX_TOKENS, 128
        )

    @classmethod
    def _fallback_model(cls) -> str:
        """Model used once when the primary model still exhausts its budget.

        Read from ``LLM_NOTES_FALLBACK_MODEL`` (e.g. in ``.env``); there is no
        hardcoded default. Unset or empty disables the fallback.
        """
        return os.getenv("LLM_NOTES_FALLBACK_MODEL", "").strip()

    @classmethod
    def _fallback_max_tokens(cls) -> int:
        return cls._positive_int_env(
            "LLM_NOTES_FALLBACK_MAX_TOKENS", cls.DEFAULT_FALLBACK_MAX_TOKENS, 128
        )

    @classmethod
    def _length_retry_enabled(cls) -> bool:
        raw_value = os.getenv("LLM_NOTES_LENGTH_RETRY_ENABLED", "").strip().lower()
        if not raw_value:
            return cls.DEFAULT_LENGTH_RETRY_ENABLED
        return raw_value in {"1", "true", "yes", "on"}

    @classmethod
    def _length_retry_token_multiplier(cls) -> float:
        raw_value = os.getenv("LLM_NOTES_LENGTH_RETRY_TOKEN_MULTIPLIER", "").strip()
        try:
            value = float(raw_value) if raw_value else cls.DEFAULT_LENGTH_RETRY_TOKEN_MULTIPLIER
        except ValueError:
            value = cls.DEFAULT_LENGTH_RETRY_TOKEN_MULTIPLIER
        return max(1.0, value)

    def _call_llm_with_length_retry(
        self,
        messages: list,
        *,
        stage: str,
        max_tokens: int,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str | None = None,
        retry_on_length: bool = True,
        allow_fallback_model: bool = True,
        events: list[str] | None = None,
    ) -> str:
        """Call the LLM with a bounded recovery chain for output-limit failures.

        Only ``LLMOutputLimitError`` (no visible content, finish_reason=length)
        triggers recovery; any other error propagates unchanged.

        1. Primary model, requested budget.
        2. Primary model once more with a larger, capped budget.
        3. Configured fallback model (default ``gpt-4o``) once.

        Human-readable notes about any recovery taken are appended to
        ``events`` so they can be surfaced in the generated notes.
        """
        call_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "system_prompt": system_prompt,
            "stage": stage,
        }
        try:
            return self.llm_service.call_llm(
                messages,
                model=model,
                max_tokens=max_tokens,
                attempt=1,
                **call_kwargs,
            )
        except LLMOutputLimitError as first_error:
            last_error: LLMOutputLimitError = first_error

        attempt = 1
        if retry_on_length and self._length_retry_enabled():
            retry_max_tokens = max(
                max_tokens + 1,
                math.ceil(max_tokens * self._length_retry_token_multiplier()),
            )
            retry_max_tokens = min(retry_max_tokens, max(self._length_retry_max_tokens(), max_tokens))
            if retry_max_tokens > max_tokens:
                attempt += 1
                logger.warning(
                    "Retrying one LLM call after output limit: stage=%s model=%s "
                    "initial_max_tokens=%d retry_max_tokens=%d",
                    stage,
                    model,
                    max_tokens,
                    retry_max_tokens,
                )
                try:
                    return self.llm_service.call_llm(
                        messages,
                        model=model,
                        max_tokens=retry_max_tokens,
                        attempt=attempt,
                        **call_kwargs,
                    )
                except LLMOutputLimitError as retry_error:
                    last_error = retry_error

        fallback_model = self._fallback_model()
        if not allow_fallback_model or not fallback_model or fallback_model == model:
            raise last_error

        fallback_max_tokens = min(max_tokens, self._fallback_max_tokens())
        logger.warning(
            "Primary model exhausted its output budget; falling back once: stage=%s "
            "primary_model=%s fallback_model=%s fallback_max_tokens=%d",
            stage,
            model,
            fallback_model,
            fallback_max_tokens,
        )
        try:
            result = self.llm_service.call_llm(
                messages,
                model=fallback_model,
                max_tokens=fallback_max_tokens,
                attempt=attempt + 1,
                **call_kwargs,
            )
        except Exception as fallback_error:
            logger.warning(
                "Fallback model call failed: stage=%s fallback_model=%s error=%s",
                stage,
                fallback_model,
                fallback_error,
            )
            raise last_error from fallback_error

        if events is not None:
            events.append(
                f"{stage}: '{model}' exhausted its output budget; "
                f"output generated by fallback model '{fallback_model}'."
            )
        return result

    @classmethod
    def _clip_prompt_text(cls, text: str, apply_char_limit: bool) -> str:
        if not apply_char_limit:
            return text
        return text[: cls._direct_transcript_char_limit()]

    @classmethod
    def _build_text_only_user_message(
        cls,
        info: dict,
        coverage: float,
        frames: list[dict],
        transcript: str,
        *,
        transcript_source_label: str = "transcript",
        apply_char_limit: bool = True,
    ) -> str:
        frame_lines = []
        for i, frame in enumerate(frames, start=1):
            line = f"[Screenshot {i} at {frame['time_str']}]: Captured frame."
            ocr_text = (frame.get("ocr_text") or "").strip()
            if ocr_text:
                line += f" OCR text hint: {ocr_text}"
            frame_lines.append(line)
        frame_descriptions = "\n".join(frame_lines)
        transcript_body = cls._clip_prompt_text(transcript, apply_char_limit=apply_char_limit)
        return textwrap.dedent(
            f"""
            Recording: {info['filename']}
            Duration: {info['duration_str']} | Size: {info['size_mb']} MB | Resolution: {info['resolution']}
            {len(frames)} screenshots at: {', '.join(f['time_str'] for f in frames)}
            Transcript coverage: {coverage:.0%}

            {frame_descriptions}

            Use {transcript_source_label} as primary source of truth for spoken content:
            {transcript_body}

            Generate comprehensive meeting notes:
            1. Meeting metadata
            2. Executive summary
            3. Key discussion topics
            4. Decisions made
            5. Action items with owners
            6. Next steps
            7. Limitations (if transcript was truncated)

            Insert @@SCREENSHOT_N@@ after the most relevant section for each screenshot.
            """
        ).strip()

    @classmethod
    def _build_multimodal_user_content(
        cls,
        info: dict,
        coverage: float,
        frames: list[dict],
        transcript: str,
        *,
        transcript_source_label: str = "transcript",
        apply_char_limit: bool = True,
    ) -> list[dict]:
        transcript_body = cls._clip_prompt_text(transcript, apply_char_limit=apply_char_limit)
        prompt_text = textwrap.dedent(
            f"""
            Recording: {info['filename']}
            Duration: {info['duration_str']} | Size: {info['size_mb']} MB | Resolution: {info['resolution']}
            {len(frames)} screenshots at: {', '.join(f['time_str'] for f in frames)}
            Transcript coverage: {coverage:.0%}

            Use {transcript_source_label} as primary source of truth for spoken content.
            Use attached screenshots as visual evidence only when they clearly match a topic.
            If a screenshot does not clearly align with a section, skip that screenshot placeholder.

            Transcript evidence:
            {transcript_body}

            Generate comprehensive meeting notes:
            1. Meeting metadata
            2. Executive summary
            3. Key discussion topics
            4. Decisions made
            5. Action items with owners
            6. Next steps
            7. Limitations (if transcript was truncated)

            Screenshot placement rules:
            - Insert @@SCREENSHOT_N@@ on its own line only where screenshot N supports the nearby text.
            - Never place placeholders in a batch without relevance.
            - Never invent visual details that are not visible in the screenshot.
            """
        ).strip()

        content: list[dict] = [{"type": "text", "text": prompt_text}]
        for i, frame in enumerate(frames, start=1):
            ocr_text = (frame.get("ocr_text") or "").strip()
            ocr_hint = f" OCR hint: {ocr_text}" if ocr_text else ""
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Screenshot {i} (timestamp {frame['time_str']}): "
                        "use @@SCREENSHOT_" + str(i) + "@@ only if this image supports the section content."
                        + ocr_hint
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame['base64_jpeg']}"},
                }
            )
        return content
    
    @classmethod
    def _build_text_only_user_message_for_transcript_only(
        cls,
        info: dict,
        coverage: float,
        transcript: str,
        *,
        transcript_source_label: str = "transcript",
        apply_char_limit: bool = True,
    ) -> str:
        transcript_body = cls._clip_prompt_text(transcript, apply_char_limit=apply_char_limit)
        return textwrap.dedent(
            f"""
            Recording: {info['filename']}
            Duration: {info['duration_str']} | Size: {info['size_mb']} MB
            Transcript coverage: {coverage:.0%}

            Use {transcript_source_label} as the only source of truth for spoken content:
            {transcript_body}

            Generate comprehensive meeting notes:
            1. Meeting metadata
            2. Executive summary
            3. Key discussion topics
            4. Decisions made
            5. Action items with owners
            6. Next steps
            7. Limitations (if transcript was truncated)
            """
        ).strip()

    @staticmethod
    def _split_transcript_into_blocks(transcript: str) -> list[str]:
        text = (transcript or "").strip()
        if not text:
            return []

        marker_regex = re.compile(r"(?m)^\[Chunk\s+\d+\]\s*$")
        markers = list(marker_regex.finditer(text))
        if not markers:
            return [text]

        blocks: list[str] = []
        for i, marker in enumerate(markers):
            start = marker.start()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            block = text[start:end].strip()
            if block:
                blocks.append(block)
        return blocks

    @staticmethod
    def _split_large_block(block: str, max_chars: int) -> list[str]:
        if len(block) <= max_chars:
            return [block]

        slices: list[str] = []
        cursor = 0
        while cursor < len(block):
            limit = min(cursor + max_chars, len(block))
            if limit < len(block):
                split_at = block.rfind("\n\n", cursor, limit)
                if split_at <= cursor:
                    split_at = block.rfind(". ", cursor, limit)
                if split_at <= cursor:
                    split_at = limit
            else:
                split_at = limit
            part = block[cursor:split_at].strip()
            if part:
                slices.append(part)
            cursor = split_at
        return slices

    @classmethod
    def _group_transcript_blocks(cls, blocks: list[str], max_chars: int) -> list[str]:
        groups: list[str] = []
        current: list[str] = []
        current_size = 0

        for block in blocks:
            for piece in cls._split_large_block(block, max_chars):
                piece_size = len(piece)
                if current and current_size + piece_size > max_chars:
                    groups.append("\n\n".join(current).strip())
                    current = []
                    current_size = 0
                current.append(piece)
                current_size += piece_size

        if current:
            groups.append("\n\n".join(current).strip())
        return groups

    SEGMENT_SYSTEM_PROMPT = (
        "You are a transcript segment analyst. "
        "Extract only explicit facts from the provided segment. "
        "Output ONLY the segment summary — no reasoning, commentary, or planning."
    )

    @staticmethod
    def _build_segment_prompt(
        segment_label: str, total_segments: int, segment_text: str, output_language: str
    ) -> str:
        return textwrap.dedent(
            f"""
            Segment {segment_label}/{total_segments}

            Transcript segment:
            {segment_text}

            Produce a concise factual summary in {output_language} using this structure:
            ### Segment {segment_label}
            - Topics discussed:
            - Decisions:
            - Action items (owner -> task):
            - Open questions / risks:
            - End-of-segment continuity notes:

            Rules:
            - Use only details present in this segment.
            - Do not infer or invent missing details.
            - If owner is unknown, write "Unknown owner".
            """
        ).strip()

    @staticmethod
    def _split_segment_in_half(segment_text: str) -> list[str]:
        """Split a segment near its midpoint on a paragraph/sentence boundary."""
        text = segment_text.strip()
        if len(text) < 2:
            return [text]
        mid = len(text) // 2
        window = max(1, len(text) // 4)
        split_at = -1
        for separator in ("\n\n", "\n", ". "):
            left = text.rfind(separator, mid - window, mid + 1)
            right = text.find(separator, mid, mid + window)
            candidates = [pos for pos in (left, right) if pos > 0]
            if candidates:
                split_at = min(candidates, key=lambda pos: abs(pos - mid)) + len(separator)
                break
        if split_at <= 0 or split_at >= len(text):
            split_at = mid
        halves = [text[:split_at].strip(), text[split_at:].strip()]
        return [half for half in halves if half]

    def _summarize_segment_resilient(
        self,
        *,
        segment_index: int,
        total_segments: int,
        segment_text: str,
        output_language: str,
        segment_max_tokens: int,
        call_kwargs: dict,
        events: list[str],
    ) -> str:
        """Summarise one segment without letting one bad segment abort the run.

        Chain: primary (+1 bounded retry) -> split into halves on the primary
        model -> fallback model per half -> raw transcript text for the segment.
        """
        prompt = self._build_segment_prompt(
            str(segment_index), total_segments, segment_text, output_language
        )
        try:
            return self._call_llm_with_length_retry(
                [{"role": "user", "content": prompt}],
                max_tokens=segment_max_tokens,
                stage="segment_summary",
                system_prompt=self.SEGMENT_SYSTEM_PROMPT,
                allow_fallback_model=False,
                events=events,
                **call_kwargs,
            ).strip()
        except LLMOutputLimitError as exc:
            logger.warning(
                "Segment %d/%d still hit the output limit; splitting into halves: %s",
                segment_index,
                total_segments,
                exc,
            )

        halves = self._split_segment_in_half(segment_text)
        half_summaries: list[str] = []
        try:
            for half_index, half_text in enumerate(halves):
                label = f"{segment_index}{chr(ord('a') + half_index)}"
                half_prompt = self._build_segment_prompt(
                    label, total_segments, half_text, output_language
                )
                half_summaries.append(
                    self._call_llm_with_length_retry(
                        [{"role": "user", "content": half_prompt}],
                        max_tokens=segment_max_tokens,
                        stage="segment_summary_split",
                        system_prompt=self.SEGMENT_SYSTEM_PROMPT,
                        retry_on_length=False,
                        allow_fallback_model=True,
                        events=events,
                        **call_kwargs,
                    ).strip()
                )
            events.append(
                f"segment {segment_index}: exceeded the output budget and was summarised in "
                f"{len(halves)} smaller parts."
            )
            return "\n\n".join(half_summaries)
        except LLMOutputLimitError as exc:
            logger.warning(
                "Segment %d/%d could not be summarised; using raw transcript text: %s",
                segment_index,
                total_segments,
                exc,
            )

        events.append(
            f"segment {segment_index}: no summary could be generated within output limits; "
            "raw transcript text was passed to the merge step instead."
        )
        return (
            f"### Segment {segment_index} (raw transcript – summary unavailable)\n"
            f"{segment_text.strip()}"
        )

    def _build_transcript_digest(
        self,
        transcript: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        output_language: str,
        events: list[str] | None = None,
    ) -> tuple[str, int, bool]:
        if events is None:
            events = []
        blocks = self._split_transcript_into_blocks(transcript)
        if not blocks:
            return transcript, 1, False

        segment_max_chars = self._map_reduce_segment_chars()
        segments = self._group_transcript_blocks(blocks, segment_max_chars)
        if not segments:
            return transcript, 1, False

        total_segments = len(segments)
        segment_max_tokens = self._segment_max_tokens(model)
        reduce_max_tokens = self._reduce_max_tokens(model)
        segment_prompts = [
            self._build_segment_prompt(
                str(segment_index), total_segments, segment_text, output_language
            )
            for segment_index, segment_text in enumerate(segments, start=1)
        ]
        call_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }

        segment_summaries: list[str] = []
        used_batch_map_reduce = False
        batch_map_error: Exception | None = None
        if self._use_batch_map_reduce():
            try:
                segment_summaries = self.llm_service.call_llm_batch_chat(
                    messages_list=[
                        [{"role": "user", "content": segment_prompt}]
                        for segment_prompt in segment_prompts
                    ],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_tokens=segment_max_tokens,
                    stage="segment_summary_batch",
                    system_prompt=self.SEGMENT_SYSTEM_PROMPT,
                )
                segment_summaries = [summary.strip() for summary in segment_summaries]
                used_batch_map_reduce = True
            except Exception as e:
                batch_map_error = e

        if not segment_summaries:
            try:
                for segment_index, segment_text in enumerate(segments, start=1):
                    segment_summaries.append(
                        self._summarize_segment_resilient(
                            segment_index=segment_index,
                            total_segments=total_segments,
                            segment_text=segment_text,
                            output_language=output_language,
                            segment_max_tokens=segment_max_tokens,
                            call_kwargs=call_kwargs,
                            events=events,
                        )
                    )
            except Exception as sync_error:
                if batch_map_error is not None:
                    raise RuntimeError(
                        "Transcript map-reduce failed in both batch mode and sync fallback. "
                        f"Batch error: {batch_map_error}"
                    ) from sync_error
                raise

        combined_summaries = "\n\n".join(segment_summaries)
        reduce_prompt = textwrap.dedent(
            f"""
            Merge these segment summaries into one consolidated transcript digest in {output_language}.

            Segment summaries:
            {combined_summaries}

            Required structure:
            ## Condensed Transcript Digest
            ### Chronological Flow
            ### Decisions
            ### Action Items (owner -> task)
            ### Open Questions / Risks
            ### Coverage Notes

            Rules:
            - Deduplicate repeated statements.
            - Keep chronology intact across the meeting.
            - Preserve caveats if details are uncertain or incomplete.
            - Do not invent attendees, decisions, or actions.
            - Some segments may be marked "raw transcript – summary unavailable";
              extract their facts directly and mention them under Coverage Notes.
            """
        ).strip()
        digest_batch_error: Exception | None = None
        if used_batch_map_reduce:
            try:
                digest = self.llm_service.call_llm_batch_chat(
                    messages_list=[[{"role": "user", "content": reduce_prompt}]],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_tokens=reduce_max_tokens,
                    stage="reduce_summary_batch",
                    system_prompt=(
                        "You are a meeting transcript synthesizer. "
                        "Produce one accurate digest from multiple segment summaries. "
                        "Output ONLY the digest — no reasoning, commentary, or planning."
                    ),
                )[0]
                return digest.strip(), total_segments, True
            except Exception as e:
                digest_batch_error = e
                used_batch_map_reduce = False

        try:
            digest = self._call_llm_with_length_retry(
                [{"role": "user", "content": reduce_prompt}],
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_tokens=reduce_max_tokens,
                stage="reduce_summary",
                system_prompt=(
                    "You are a meeting transcript synthesizer. "
                    "Produce one accurate digest from multiple segment summaries. "
                    "Output ONLY the digest — no reasoning, commentary, or planning."
                ),
                events=events,
            )
            return digest.strip(), total_segments, False
        except Exception as sync_error:
            if digest_batch_error is not None:
                raise RuntimeError(
                    "Transcript reduce-step failed in both batch mode and sync fallback. "
                    f"Batch error: {digest_batch_error}"
                ) from sync_error
            raise

    @classmethod
    def _chat_request_budget_bytes(cls) -> int:
        raw_limit = os.getenv("LLM_MAX_CHAT_REQUEST_BYTES", "").strip()
        try:
            limit = int(raw_limit) if raw_limit else cls.DEFAULT_CHAT_REQUEST_LIMIT_BYTES
        except ValueError:
            limit = cls.DEFAULT_CHAT_REQUEST_LIMIT_BYTES
        return max(1, int(limit * cls.CHAT_REQUEST_SAFETY_RATIO))

    @staticmethod
    def _resolution_pixels(resolution: str) -> int | None:
        match = re.search(r"(\d+)\s*x\s*(\d+)", resolution or "", flags=re.IGNORECASE)
        if not match:
            return None
        width = int(match.group(1))
        height = int(match.group(2))
        if width <= 0 or height <= 0:
            return None
        return width * height

    @classmethod
    def _estimated_screenshot_request_bytes(
        cls,
        resolution: str,
        avg_frame_request_bytes: int | None = None,
    ) -> int:
        if avg_frame_request_bytes and avg_frame_request_bytes > 0:
            return int(avg_frame_request_bytes)

        pixels = cls._resolution_pixels(resolution)
        if not pixels:
            return cls.DEFAULT_SCREENSHOT_REQUEST_BYTES

        megapixels = pixels / 1_000_000
        return max(cls.DEFAULT_SCREENSHOT_REQUEST_BYTES, int(megapixels * 220_000))

    @classmethod
    def recommend_capture_settings(
        cls,
        *,
        duration_seconds: float,
        resolution: str,
        avg_frame_request_bytes: int | None = None,
    ) -> dict:
        duration = max(1.0, float(duration_seconds or 0))
        budget_bytes = cls._chat_request_budget_bytes()
        reserve_bytes = min(cls.CHAT_REQUEST_TEXT_RESERVE_BYTES, max(0, budget_bytes - 1))
        frame_request_bytes = cls._estimated_screenshot_request_bytes(
            resolution,
            avg_frame_request_bytes,
        )

        available_for_frames = max(1, budget_bytes - reserve_bytes)
        max_frames_by_budget = max(1, available_for_frames // max(1, frame_request_bytes))
        target_frames_by_duration = max(1, min(120, round(duration / 60)))
        max_frames = max(1, min(120, target_frames_by_duration, max_frames_by_budget))

        estimated_request_bytes = reserve_bytes + (max_frames * frame_request_bytes)
        while estimated_request_bytes > budget_bytes and max_frames > 1:
            max_frames -= 1
            estimated_request_bytes = reserve_bytes + (max_frames * frame_request_bytes)

        raw_interval = duration / (max_frames + 1)
        interval_seconds = int(math.floor(raw_interval / 5) * 5)
        interval_seconds = max(10, min(600, interval_seconds))

        return {
            "interval_seconds": interval_seconds,
            "max_frames": int(max_frames),
            "estimated_request_bytes": int(estimated_request_bytes),
            "budget_bytes": int(budget_bytes),
            "frame_request_bytes": int(frame_request_bytes),
        }

    @classmethod
    def _estimate_chat_request_bytes(
        cls,
        messages: list,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> int:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": full_messages,
        }
        return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    @classmethod
    def _chat_request_exceeds_budget(
        cls,
        messages: list,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> bool:
        return cls._estimate_chat_request_bytes(
            messages,
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        ) > cls._chat_request_budget_bytes()

    @staticmethod
    def _is_request_too_large_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        hints = (
            "content length exceeded",
            "request too large",
            "payload too large",
            "413",
        )
        return any(hint in msg for hint in hints)

    @staticmethod
    def _is_vision_unsupported_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        hints = (
            "image_url",
            "vision",
            "does not support image",
            "content type",
            "invalid image",
            "invalid chat format",
        )
        return any(hint in msg for hint in hints)

    def _ocr_with_document_model(
        self,
        frame: dict,
        *,
        base_url: str,
        api_key: str,
        ocr_model: str,
    ) -> str:
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract the visible on-screen text from this screenshot. "
                        "Return plain text only. If no readable text exists, return NO_TEXT."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame['base64_jpeg']}"},
                },
            ],
        }
        extracted = self.llm_service.call_llm(
            [message],
            base_url=base_url,
            api_key=api_key,
            model=ocr_model,
            max_tokens=600,
            system_prompt=(
                "You are an OCR extraction assistant. "
                "Do not summarize. Output only text visible in the image."
            ),
        )
        normalized = " ".join((extracted or "").split())
        if normalized.upper().startswith("NO_TEXT"):
            return ""
        if len(normalized) > 300:
            normalized = normalized[:300] + "..."
        return normalized

    def _enrich_frames_with_document_model_ocr(
        self,
        frames: list[dict],
        *,
        base_url: str,
        api_key: str,
    ) -> dict:
        ocr_model = os.getenv("LLM_OCR_MODEL", "").strip()
        if not ocr_model:
            logger.info("LLM_OCR_MODEL is not set; OCR is disabled by default.")
            for frame in frames:
                frame["ocr_text"] = ""
                frame["ocr_keywords"] = []
            return {"available": False, "engine": "none", "reason": "disabled by default"}

        logger.info(
            "Starting document model OCR with model '%s' across %d frame(s) (base_url: %s)...",
            ocr_model,
            len(frames),
            base_url,
        )
        extracted_count = 0
        failures = 0
        last_error = ""
        for frame in frames:
            frame["ocr_text"] = ""
            frame["ocr_keywords"] = []

        def _ocr_single(i: int, frame: dict) -> tuple[int, str | None, Exception | None]:
            try:
                res_text = self._ocr_with_document_model(
                    frame,
                    base_url=base_url,
                    api_key=api_key,
                    ocr_model=ocr_model,
                )
                return i, res_text, None
            except Exception as exc:
                return i, None, exc

        max_workers = min(self._ocr_max_workers(), max(1, len(frames)))
        ocr_outcomes: dict[int, tuple[str | None, Exception | None]] = {}
        if max_workers <= 1 or len(frames) <= 1:
            for i, frame in enumerate(frames, start=1):
                _, res_text, exc = _ocr_single(i, frame)
                ocr_outcomes[i] = (res_text, exc)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(copy_context().run, _ocr_single, i, frame): i
                    for i, frame in enumerate(frames, start=1)
                }
                for future in as_completed(futures):
                    i, res_text, exc = future.result()
                    ocr_outcomes[i] = (res_text, exc)

        for i, frame in enumerate(frames, start=1):
            text, exc = ocr_outcomes[i]
            if exc is not None:
                failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Document model OCR failed on frame %d/%d (%s) with model '%s': %s: %s",
                    i,
                    len(frames),
                    frame.get("time_str", "N/A"),
                    ocr_model,
                    type(exc).__name__,
                    exc,
                )
                continue

            text = text or ""
            if text:
                logger.debug(
                    "Document model OCR extracted text from frame %d/%d (%s): %s",
                    i,
                    len(frames),
                    frame.get("time_str", "N/A"),
                    text[:80] + ("..." if len(text) > 80 else ""),
                )
            else:
                logger.debug(
                    "Document model OCR returned empty/no text for frame %d/%d (%s)",
                    i,
                    len(frames),
                    frame.get("time_str", "N/A"),
                )

            frame["ocr_text"] = text
            frame["ocr_keywords"] = list(self._extract_keywords(text)) if text else []
            if text:
                extracted_count += 1

        logger.info(
            "Document model OCR completed: %d/%d frame(s) with text, %d failure(s).",
            extracted_count,
            len(frames),
            failures,
        )

        if extracted_count == 0 and failures > 0:
            reason = f"document model OCR failed for {failures}/{len(frames)} frame(s)"
            if last_error:
                reason += f" ({last_error})"
            return {
                "available": False,
                "engine": ocr_model,
                "reason": reason,
            }

        return {
            "available": True,
            "engine": ocr_model,
            "frames_with_text": extracted_count,
            "total_frames": len(frames),
        }

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        stop_words = {
            "this", "that", "with", "from", "have", "will", "your", "about", "there",
            "their", "into", "after", "before", "would", "could", "should", "were",
            "been", "being", "also", "only", "when", "what", "where", "which",
            "while", "than", "then", "them", "they", "over", "under", "more", "most",
        }
        tokens = re.findall(r"[a-zA-Z0-9]{4,}", text.lower())
        return {t for t in tokens if t not in stop_words}

    def _build_alignment_warnings(self, notes_text: str, frames: list[dict]) -> list[str]:
        warnings: list[str] = []
        for i, frame in enumerate(frames, start=1):
            placeholder = f"@@SCREENSHOT_{i}@@"
            pos = notes_text.find(placeholder)
            if pos < 0:
                continue

            ocr_text = (frame.get("ocr_text") or "").strip()
            ocr_keywords = set(frame.get("ocr_keywords") or []) or self._extract_keywords(ocr_text)
            if not ocr_keywords:
                continue

            window = notes_text[max(0, pos - 450): min(len(notes_text), pos + 450)]
            section_keywords = self._extract_keywords(window)
            if not section_keywords:
                continue

            overlap = section_keywords.intersection(ocr_keywords)
            score = len(overlap) / max(1, min(len(section_keywords), len(ocr_keywords)))
            if score < 0.08:
                warnings.append(
                    f"Screenshot {i} at {frame['time_str']} may be weakly aligned with nearby section text "
                    f"(keyword overlap {score:.0%})."
                )
        return warnings

    @staticmethod
    def build_meeting_context(
        *,
        info: dict,
        markdown_notes: str,
        transcript_evidence: str,
        transcript_source_label: str,
        frames: list[dict],
        output_language: str,
        notes_style: str,
    ) -> MeetingContext:
        """Build a lightweight, JSON-serializable MeetingContext without PIL or base64 blobs."""
        clean_info = {
            "filename": str(info.get("filename", "")),
            "duration": float(info.get("duration", 0.0) or 0.0),
            "duration_str": str(info.get("duration_str", "N/A")),
            "size_mb": str(info.get("size_mb", "0.00")),
            "resolution": str(info.get("resolution", "N/A")),
        }
        frames_metadata = [
            {
                "index": idx,
                "time_str": str(frame.get("time_str", "")),
                "ocr_text": (frame.get("ocr_text") or "").strip(),
                "ocr_keywords": list(frame.get("ocr_keywords") or []),
            }
            for idx, frame in enumerate(frames or [], start=1)
        ]
        return MeetingContext(
            info=clean_info,
            markdown_notes=(markdown_notes or "").strip(),
            transcript_evidence=(transcript_evidence or "").strip(),
            transcript_source_label=str(transcript_source_label or "transcript"),
            frames_metadata=frames_metadata,
            output_language=str(output_language or "English"),
            notes_style=str(notes_style or "Executive Summary"),
        )

    def run_pipeline(
        self,
        video_file,
        base_url,
        api_key,
        model,
        interval_seconds,
        max_frames,
        notes_style,
        output_language,
        transcript_file=None,
        reasoning_effort=None,
        progress=gr.Progress(track_tqdm=False),
        *,
        return_context: bool = False,
        include_context: bool = False,
    ):
        """Run the pipeline with an optional per-run reasoning effort.

        ``reasoning_effort``: ``None`` = use env config; ``"none"`` = send no
        reasoning setting; otherwise ``{"reasoning": {"effort": X}}`` is sent
        and silently dropped if the model rejects it.
        """
        with self.llm_service.reasoning_effort(reasoning_effort):
            result = self._run_pipeline_impl(
                video_file,
                base_url,
                api_key,
                model,
                interval_seconds,
                max_frames,
                notes_style,
                output_language,
                transcript_file=transcript_file,
                progress=progress,
            )
        if return_context or include_context:
            thumbnails, notes_text, html_notes, out_path = result
            return thumbnails, notes_text, html_notes, out_path, result.meeting_context
        return result

    def run_pipeline_with_context(
        self,
        video_file,
        base_url,
        api_key,
        model,
        interval_seconds,
        max_frames,
        notes_style,
        output_language,
        transcript_file=None,
        reasoning_effort=None,
        progress=gr.Progress(track_tqdm=False),
    ):
        """Run the pipeline and return the 5-tuple including ``MeetingContext``."""
        return self.run_pipeline(
            video_file,
            base_url,
            api_key,
            model,
            interval_seconds,
            max_frames,
            notes_style,
            output_language,
            transcript_file=transcript_file,
            reasoning_effort=reasoning_effort,
            progress=progress,
            return_context=True,
        )

    def _run_pipeline_impl(
        self,
        video_file,
        base_url,
        api_key,
        model,
        interval_seconds,
        max_frames,
        notes_style,
        output_language,
        transcript_file=None,
        progress=gr.Progress(track_tqdm=False),
    ):
        if video_file is None and transcript_file is None:
            raise gr.Error("Please upload a video file or a transcript file first.")
        if not base_url.strip():
            raise gr.Error("LLM base URL is required.")
        if not model.strip():
            raise gr.Error("Model name is required.")

        is_transcript_only = transcript_file is not None and video_file is None
        staged_video_path: Path | None = None

        if is_transcript_only:
            progress(0.10, desc="Reading uploaded transcript...")
            transcript_path, saved_transcript = self._load_transcript_text(transcript_file)
            transcript_size_mb = transcript_path.stat().st_size / 1_048_576
            info = {
                "filename": transcript_path.name,
                "duration": 0.0,
                "duration_str": "N/A",
                "size_mb": f"{transcript_size_mb:.2f}",
                "resolution": "N/A",
            }
            tx = {
                "transcript_path": str(transcript_path),
                "transcript_plain_path": str(transcript_path),
                "manifest_path": "N/A (user-provided transcript)",
                "manifest": {"coverage_ratio": 1.0, "status": "user_provided"},
            }
            coverage = 1.0
            frames = []
            thumbnails = []
            ocr_meta = {
                "available": False,
                "engine": "none",
                "reason": "Transcript-only input",
            }
        else:
            source_video_path = self._resolve_input_video_path(video_file)
            if not source_video_path.exists():
                raise gr.Error(f"Uploaded video path does not exist: {source_video_path}")

            progress(0.03, desc="Preparing uploaded video...")
            try:
                staged_video_path, _duplicate_info = self._stage_video_copy(source_video_path)
            except PermissionError as e:
                raise gr.Error(
                    "Cannot read the uploaded video from Gradio temporary storage on Windows. "
                    "Please re-upload the file and try again."
                ) from e

            video_path = str(staged_video_path)

            progress(0.05, desc="Reading video metadata...")
            info = self.video_service.get_video_info(video_path)

            progress(
                0.20,
                desc=f"Extracting frames ({interval_seconds}s) & transcribing audio in parallel...",
            )

            def _visual_branch() -> tuple[list[dict], dict, list]:
                extracted_frames = self.video_service.extract_frames(
                    video_path, int(interval_seconds), int(max_frames)
                )
                meta = self._enrich_frames_with_document_model_ocr(
                    extracted_frames,
                    base_url=base_url,
                    api_key=api_key,
                )
                if meta.get("engine") != "none" and (
                    (not meta.get("available", False)) or meta.get("frames_with_text", 0) == 0
                ):
                    fallback_meta = self.video_service.enrich_frames_with_ocr(extracted_frames)
                    if fallback_meta.get("available", False):
                        meta = fallback_meta
                    elif not meta.get("available", False):
                        reason_parts = [meta.get("reason", ""), fallback_meta.get("reason", "")]
                        merged_reason = "; ".join(part for part in reason_parts if part)
                        meta = {"available": False, "engine": "none", "reason": merged_reason}

                thumbs = [f["pil_image"] for f in extracted_frames]
                return extracted_frames, meta, thumbs

            transcribe_model = (
                os.getenv("LLM_TRANSCRIBE_MODEL", "").strip()
                or "gemini-3.5-transcribe"
            )
            transcribe_base_url = os.getenv("LLM_TRANSCRIBE_BASE_URL", "").strip()
            self.video_service.validate_transcription_config(transcribe_model, transcribe_base_url)

            def _audio_branch() -> dict:
                if self.video_service.is_gemini_transcribe_model(transcribe_model):
                    transcribe_api_key = (
                        os.getenv("LLM_TRANSCRIBE_API_KEY", "").strip()
                        or os.getenv("GEMINI_API_KEY", "").strip()
                        or os.getenv("GOOGLE_API_KEY", "").strip()
                        or api_key
                    )
                else:
                    transcribe_api_key = os.getenv("LLM_TRANSCRIBE_API_KEY", "").strip() or api_key
                return self.video_service.transcribe_video_chunked(
                    video_path,
                    base_url=transcribe_base_url,
                    api_key=transcribe_api_key,
                    transcribe_model=transcribe_model,
                    chunk_seconds=90,
                    chunk_overlap_seconds=10,
                )

            with ThreadPoolExecutor(max_workers=2) as branch_executor:
                visual_future = branch_executor.submit(copy_context().run, _visual_branch)
                audio_future = branch_executor.submit(copy_context().run, _audio_branch)
                frames, ocr_meta, thumbnails = visual_future.result()
                tx = audio_future.result()

            saved_transcript = tx["transcript_text"]
            coverage = tx["manifest"].get("coverage_ratio", 0.0)

        progress(0.62, desc=f"Calling {model}...")

        style = self.notes_service.STYLE_GUIDES.get(
            notes_style,
            self.notes_service.STYLE_GUIDES["Executive Summary"],
        )

        local_tz = self._detect_local_timezone()

        system_prompt = textwrap.dedent(
            f"""
            You are an expert meeting notes assistant.
            {style}
            Output in {output_language}. Use Markdown.
            The recording was made in the {local_tz} timezone. Use this timezone when
            noting meeting times — do not guess the timezone from participant names or
            meeting content.
            Prioritize transcript fidelity over visual guessing.
            Only insert @@SCREENSHOT_N@@ when screenshot N clearly supports nearby text.
            Do not force placeholders for every section.
            CRITICAL: Output ONLY the final meeting notes. Do NOT include any
            chain-of-thought, reasoning, planning, commentary, or internal monologue.
            """
        ).strip()

        transcript_source_label = "transcript"
        transcript_evidence = saved_transcript
        used_map_reduce_transcript = False
        map_reduce_segments = 0
        used_batch_map_reduce = False
        llm_recovery_events: list[str] = []
        if len(saved_transcript) > self._map_reduce_trigger_chars():
            progress(0.57, desc="Condensing long transcript in segments...")
            transcript_evidence, map_reduce_segments, used_batch_map_reduce = self._build_transcript_digest(
                saved_transcript,
                base_url=base_url,
                api_key=api_key,
                model=model,
                output_language=output_language,
                events=llm_recovery_events,
            )
            used_map_reduce_transcript = True
            transcript_source_label = (
                "condensed transcript digest generated from all transcript segments"
            )

        if is_transcript_only:
            text_message = {
                "role": "user",
                "content": self._build_text_only_user_message_for_transcript_only(
                    info,
                    coverage,
                    transcript_evidence,
                    transcript_source_label=transcript_source_label,
                    apply_char_limit=not used_map_reduce_transcript,
                ),
            }
            raw_notes = self._call_llm_with_length_retry(
                [text_message],
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_tokens=self._final_max_tokens(model),
                stage="final_notes",
                system_prompt=system_prompt,
                events=llm_recovery_events,
            )
            used_text_only_due_to_size = False
        else:
            multimodal_message = {
                "role": "user",
                "content": self._build_multimodal_user_content(
                    info,
                    coverage,
                    frames,
                    transcript_evidence,
                    transcript_source_label=transcript_source_label,
                    apply_char_limit=not used_map_reduce_transcript,
                ),
            }
            text_message = {
                "role": "user",
                "content": self._build_text_only_user_message(
                    info,
                    coverage,
                    frames,
                    transcript_evidence,
                    transcript_source_label=transcript_source_label,
                    apply_char_limit=not used_map_reduce_transcript,
                ),
            }

            notes_max_tokens = self._final_max_tokens(model)
            used_text_only_due_to_size = self._chat_request_exceeds_budget(
                [multimodal_message],
                model=model,
                max_tokens=notes_max_tokens,
                system_prompt=system_prompt,
            )

            try:
                if used_text_only_due_to_size:
                    raise ValueError("multimodal chat request would exceed configured request-size budget")
                raw_notes = self._call_llm_with_length_retry(
                    [multimodal_message],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_tokens=notes_max_tokens,
                    stage="final_notes",
                    system_prompt=system_prompt,
                    events=llm_recovery_events,
                )
            except Exception as e:
                if not (
                    used_text_only_due_to_size
                    or self._is_vision_unsupported_error(e)
                    or self._is_request_too_large_error(e)
                ):
                    raise
                used_text_only_due_to_size = used_text_only_due_to_size or self._is_request_too_large_error(e)
                raw_notes = self._call_llm_with_length_retry(
                    [text_message],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_tokens=notes_max_tokens,
                    stage="final_notes_text_fallback",
                    system_prompt=system_prompt,
                    events=llm_recovery_events,
                )

        alignment_warnings = self._build_alignment_warnings(raw_notes, frames)

        progress(0.83, desc="Building HTML output...")
        html_notes = self.notes_service.build_html_notes(raw_notes, frames, info, model)

        minutes_dir = Path.cwd() / "artifacts" / "meeting_minutes"
        minutes_dir.mkdir(parents=True, exist_ok=True)
        images_dir = minutes_dir / "images"
        minutes_path = minutes_dir / (info["filename"].rsplit(".", 1)[0] + ".minutes.md")
        markdown_body = self.notes_service.build_markdown_notes(
            raw_notes, frames, info, images_dir, minutes_dir
        )

        transcript_manifest_value = tx["manifest_path"]
        if transcript_manifest_value == "N/A (user-provided transcript)":
            transcript_manifest_display = transcript_manifest_value
        else:
            transcript_manifest_display = Path(transcript_manifest_value).name

        minutes_doc = (
            f"# Meeting Minutes\n\n"
            + (f"source_transcript: {info['filename']}\n" if is_transcript_only else f"source_video: {info['filename']}\n")
            + f"transcript_file: {Path(tx['transcript_path']).name}\n"
            + f"transcript_manifest: {transcript_manifest_display}\n"
            + f"ocr_available: {ocr_meta.get('available', False)}\n"
            + f"screenshot_interval_seconds: {interval_seconds}\n"
            + f"max_screenshots: {max_frames}\n"
            + f"screenshots_captured: {len(frames)}\n"
            + f"generated_at: {datetime.now().isoformat()}\n\n"
            + markdown_body
            + (
                "\n\n## Screenshot Alignment Checks\n\n"
                + "\n".join(f"- {w}" for w in alignment_warnings)
                if alignment_warnings
                else ""
            )
            + "\n"
        )
        minutes_path.write_text(minutes_doc, encoding="utf-8")

        if staged_video_path is not None:
            self._record_minutes_for_video(staged_video_path, minutes_path.name)

        out_name = info["filename"].rsplit(".", 1)[0] + "_meeting_notes.html"
        out_path = os.path.join(tempfile.gettempdir(), out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html_notes)

        artifact_summary = (
            "\n\n---\n"
            "Artifacts\n"
            f"- Transcript: {tx['transcript_path']}\n"
            f"- Plain transcript: {tx['transcript_plain_path']}\n"
            f"- Transcript manifest: {tx['manifest_path']}\n"
            f"- Minutes markdown: {minutes_path}\n"
            f"- Screenshots: {len(frames)} captured (interval {interval_seconds}s, max {max_frames})\n"
            f"- OCR available: {ocr_meta.get('available', False)}"
            + (
                f" ({ocr_meta.get('reason')})\n" if not ocr_meta.get("available", False) else "\n"
            )
            + (
                "- Screenshot alignment warnings:\n"
                + "\n".join(f"  - {w}" for w in alignment_warnings)
                + "\n"
                if alignment_warnings
                else ""
            )
            + (
                "- Note: screenshot images were omitted from the notes-generation request because "
                "the estimated HTTP body would exceed the configured request-size limit; timestamp and OCR hints were used instead.\n"
                if used_text_only_due_to_size
                else ""
            )
            + (
                "- Note: transcript-only mode was used; video, screenshots, OCR, and audio transcription were skipped.\n"
                if is_transcript_only
                else ""
            )
            + (
                "- Note: long-transcript map-reduce summarization was applied across "
                f"{map_reduce_segments} segment(s) before final note generation"
                + (" using batch mode.\n" if used_batch_map_reduce else " using synchronous mode.\n")
                if used_map_reduce_transcript
                else ""
            )
            + (
                "- LLM output-limit recovery:\n"
                + "\n".join(f"  - {event}" for event in llm_recovery_events)
                + "\n"
                if llm_recovery_events
                else ""
            )
        )

        meeting_context = self.build_meeting_context(
            info=info,
            markdown_notes=markdown_body,
            transcript_evidence=transcript_evidence,
            transcript_source_label=transcript_source_label,
            frames=frames,
            output_language=output_language,
            notes_style=notes_style,
        )

        progress(1.0, desc="Done!")
        return PipelineRunResult(
            (thumbnails, raw_notes + artifact_summary, html_notes, out_path),
            meeting_context=meeting_context,
        )

    def run_pipeline_safe(
        self,
        video_file,
        base_url,
        api_key,
        model,
        interval_seconds,
        max_frames,
        notes_style,
        output_language,
        transcript_file=None,
        reasoning_effort=None,
        progress=gr.Progress(track_tqdm=False),
        *,
        return_context: bool = False,
        include_context: bool = False,
    ):
        """UI-safe wrapper: return friendly output on failure instead of raising."""
        try:
            return self.run_pipeline(
                video_file,
                base_url,
                api_key,
                model,
                interval_seconds,
                max_frames,
                notes_style,
                output_language,
                transcript_file=transcript_file,
                reasoning_effort=reasoning_effort,
                progress=progress,
                return_context=return_context,
                include_context=include_context,
            )
        except gr.Error:
            raise
        except TranscriptionConfigError as e:
            raise gr.Error(str(e), title="Configuration error") from e
        except NoAudioStreamError as e:
            # Surface as a Gradio pop-up instead of a raw ffmpeg dump in the notes pane.
            raise gr.Error(NoAudioStreamError.USER_MESSAGE, title="No audio detected") from e
        except Exception as e:
            msg = (
                "Generation failed. You can upload again and retry.\n\n"
                f"Error: {type(e).__name__}: {e}"
            )
            html = (
                "<div style='padding:12px;border:1px solid #f4c2c2;background:#fff7f7;'>"
                "<b>Generation failed.</b><br/>"
                "Please click <i>Reset</i>, upload the video again, and retry."
                "</div>"
            )
            if return_context or include_context:
                return [], msg, html, None, None
            return PipelineRunResult(([], msg, html, None), meeting_context=None)

    def run_pipeline_safe_with_context(
        self,
        video_file,
        base_url,
        api_key,
        model,
        interval_seconds,
        max_frames,
        notes_style,
        output_language,
        transcript_file=None,
        reasoning_effort=None,
        progress=gr.Progress(track_tqdm=False),
    ):
        """UI-safe wrapper returning the 5-tuple including ``MeetingContext``."""
        return self.run_pipeline_safe(
            video_file,
            base_url,
            api_key,
            model,
            interval_seconds,
            max_frames,
            notes_style,
            output_language,
            transcript_file=transcript_file,
            reasoning_effort=reasoning_effort,
            progress=progress,
            return_context=True,
        )

    @staticmethod
    def _extract_chat_message_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    text_val = item.get("text") or item.get("content")
                    if isinstance(text_val, str) and text_val.strip():
                        parts.append(text_val.strip())
            return "\n".join(parts).strip()
        if content is None:
            return ""
        return str(content).strip()

    @classmethod
    def _normalize_chat_history(cls, chat_history: list | None) -> list[dict[str, str]]:
        """Normalize Gradio Chatbot history (dict messages or tuple pairs) into role/content dicts."""
        if not chat_history:
            return []
        normalized: list[dict[str, str]] = []
        for entry in chat_history:
            if isinstance(entry, dict):
                role = str(entry.get("role", "")).strip().lower()
                content = cls._extract_chat_message_text(entry.get("content"))
                if role in ("user", "assistant") and content:
                    normalized.append({"role": role, "content": content})
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                user_text = cls._extract_chat_message_text(entry[0])
                assistant_text = cls._extract_chat_message_text(entry[1])
                if user_text:
                    normalized.append({"role": "user", "content": user_text})
                if assistant_text:
                    normalized.append({"role": "assistant", "content": assistant_text})
            elif hasattr(entry, "role") and hasattr(entry, "content"):
                role = str(getattr(entry, "role", "")).strip().lower()
                content = cls._extract_chat_message_text(getattr(entry, "content", ""))
                if role in ("user", "assistant") and content:
                    normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _is_valid_meeting_context(meeting_context: dict | None) -> bool:
        if not isinstance(meeting_context, dict) or not meeting_context:
            return False
        has_notes = bool(str(meeting_context.get("markdown_notes") or "").strip())
        has_transcript = bool(str(meeting_context.get("transcript_evidence") or "").strip())
        return has_notes or has_transcript

    @staticmethod
    def _build_qa_system_prompt(output_language: str) -> str:
        return textwrap.dedent(
            f"""
            You are a grounded meeting Q&A assistant.
            Answer questions about the meeting using ONLY the provided meeting metadata, generated meeting minutes, captured screenshot timestamps and OCR text hints, and transcript evidence.
            Rules:
            1. Prioritize the transcript evidence, OCR text hints, and generated meeting notes as your sole sources of truth.
            2. Cite or reference relevant timestamps, speakers, or screenshot numbers (e.g., "Screenshot 1 at 0:02") whenever that evidence is present in the transcript or OCR metadata.
            3. If a question cannot be answered from the provided meeting content, explicitly state that the information is not mentioned in the meeting recording or transcript rather than guessing or fabricating facts.
            4. Default to responding in {output_language} (or respond in the language of the user's question when clearly asked in another language).
            5. CRITICAL: Output ONLY the final answer. Do NOT include any chain-of-thought, reasoning, planning, commentary, or internal monologue.
            """
        ).strip()

    @classmethod
    def _build_qa_grounding_message(
        cls,
        meeting_context: dict,
        *,
        question: str,
        transcript_char_limit: int | None = None,
        notes_char_limit: int | None = None,
    ) -> str:
        info = meeting_context.get("info") if isinstance(meeting_context.get("info"), dict) else {}
        filename = info.get("filename") or "Unknown"
        duration_str = info.get("duration_str") or "N/A"
        resolution = info.get("resolution") or "N/A"
        notes_style = meeting_context.get("notes_style") or "Executive Summary"
        output_language = meeting_context.get("output_language") or "English"

        markdown_notes = str(meeting_context.get("markdown_notes") or "").strip()
        if notes_char_limit is not None and notes_char_limit >= 0:
            markdown_notes = markdown_notes[:notes_char_limit]

        transcript_source_label = (
            str(meeting_context.get("transcript_source_label") or "transcript").strip()
        )
        transcript_evidence = str(meeting_context.get("transcript_evidence") or "").strip()
        if transcript_char_limit is not None and transcript_char_limit >= 0:
            transcript_evidence = transcript_evidence[:transcript_char_limit]

        frames_metadata = meeting_context.get("frames_metadata") or []
        if frames_metadata:
            frame_lines: list[str] = []
            for i, frame in enumerate(frames_metadata, start=1):
                if not isinstance(frame, dict):
                    continue
                idx = frame.get("index", i)
                time_str = frame.get("time_str", "N/A")
                ocr_text = str(frame.get("ocr_text") or "").strip()
                keywords = frame.get("ocr_keywords") or []
                line = f"[Screenshot {idx} at {time_str}]: Captured frame."
                if ocr_text:
                    line += f" OCR text hint: {ocr_text}"
                if keywords:
                    line += f" (keywords: {', '.join(str(k) for k in keywords)})"
                frame_lines.append(line)
            frames_section = "\n".join(frame_lines) if frame_lines else "No screenshots captured."
        else:
            frames_section = "No screenshots captured (transcript-only mode)."

        return textwrap.dedent(
            f"""
            Meeting Context
            Recording: {filename}
            Duration: {duration_str} | Resolution: {resolution}
            Notes style: {notes_style} | Output language: {output_language}

            Captured Screenshots & OCR Metadata:
            {frames_section}

            Generated Meeting Minutes:
            {markdown_notes}

            Transcript Evidence ({transcript_source_label}):
            {transcript_evidence}

            Answer the user's question using the meeting context above. Cite relevant timestamps, speakers, or screenshot numbers when available, and state clearly if something was not mentioned in the meeting.
            Current question: {question}
            """
        ).strip()

    def _assemble_budgeted_qa_messages(
        self,
        *,
        meeting_context: dict,
        normalized_history: list[dict[str, str]],
        question: str,
        model: str,
        max_tokens: int,
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """Assemble grounding context, bounded history turns, and current question within byte budget."""
        budget_bytes = self._chat_request_budget_bytes()
        raw_transcript = str(meeting_context.get("transcript_evidence") or "").strip()
        raw_notes = str(meeting_context.get("markdown_notes") or "").strip()
        transcript_limit = min(len(raw_transcript), self._direct_transcript_char_limit())
        notes_limit = len(raw_notes)
        history_slice = list(normalized_history)

        def _make_messages(tx_limit: int, md_limit: int, hist: list[dict[str, str]]) -> list[dict[str, str]]:
            grounding = self._build_qa_grounding_message(
                meeting_context,
                question=question,
                transcript_char_limit=tx_limit,
                notes_char_limit=md_limit,
            )
            return [
                {"role": "user", "content": grounding},
                *hist,
                {"role": "user", "content": question},
            ]

        messages = _make_messages(transcript_limit, notes_limit, history_slice)
        if not self._chat_request_exceeds_budget(
            messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
        ):
            return messages

        # Step 1: Trim oldest conversation turns from chat_history (in pairs when possible)
        while history_slice and self._chat_request_exceeds_budget(
            messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
        ):
            if (
                len(history_slice) >= 2
                and history_slice[0]["role"] == "user"
                and history_slice[1]["role"] == "assistant"
            ):
                history_slice = history_slice[2:]
            else:
                history_slice = history_slice[1:]
            messages = _make_messages(transcript_limit, notes_limit, history_slice)

        # Step 2: Bound transcript evidence if grounding message still exceeds budget
        while transcript_limit > 200 and self._chat_request_exceeds_budget(
            messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
        ):
            next_limit = max(200, transcript_limit // 2)
            messages = _make_messages(next_limit, notes_limit, history_slice)
            if next_limit == transcript_limit:
                break
            transcript_limit = next_limit

        # Step 3: Bound markdown notes if still exceeding budget under extreme limits
        while notes_limit > 200 and self._chat_request_exceeds_budget(
            messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
        ):
            next_limit = max(200, notes_limit // 2)
            messages = _make_messages(transcript_limit, next_limit, history_slice)
            if next_limit == notes_limit:
                break
            notes_limit = next_limit

        # Step 4: If budget is still exceeded (e.g. tight budget), binary-search transcript and notes down to 0
        if self._chat_request_exceeds_budget(
            messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
        ):
            while transcript_limit > 0 and self._chat_request_exceeds_budget(
                messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
            ):
                transcript_limit = transcript_limit // 2
                messages = _make_messages(transcript_limit, notes_limit, history_slice)
            while notes_limit > 0 and self._chat_request_exceeds_budget(
                messages, model=model, max_tokens=max_tokens, system_prompt=system_prompt
            ):
                notes_limit = notes_limit // 2
                messages = _make_messages(transcript_limit, notes_limit, history_slice)

        return messages

    def query_meeting(
        self,
        question: str,
        chat_history: list | None,
        meeting_context: dict | None,
        base_url: str,
        api_key: str,
        model: str,
        output_language: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """Answer a follow-up question grounded in the active session's ``MeetingContext``."""
        normalized_history = self._normalize_chat_history(chat_history)
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            return "", normalized_history

        if not self._is_valid_meeting_context(meeting_context):
            raise gr.Error(
                "Please generate meeting notes first before asking questions about the meeting."
            )
        if not (base_url or "").strip():
            raise gr.Error("LLM base URL is required.")
        if not (model or "").strip():
            raise gr.Error("Model name is required.")

        assert isinstance(meeting_context, dict)
        effective_language = (
            (output_language or "").strip()
            or str(meeting_context.get("output_language") or "").strip()
            or "English"
        )
        system_prompt = self._build_qa_system_prompt(effective_language)
        qa_max_tokens = self._qa_max_tokens(model)
        messages = self._assemble_budgeted_qa_messages(
            meeting_context=meeting_context,
            normalized_history=normalized_history,
            question=cleaned_question,
            model=model,
            max_tokens=qa_max_tokens,
            system_prompt=system_prompt,
        )

        with self.llm_service.reasoning_effort(reasoning_effort):
            reply = self._call_llm_with_length_retry(
                messages,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_tokens=qa_max_tokens,
                stage="meeting_qa",
                system_prompt=system_prompt,
            ).strip()

        updated_history = [
            *normalized_history,
            {"role": "user", "content": cleaned_question},
            {"role": "assistant", "content": reply},
        ]
        return reply, updated_history

    def query_meeting_safe(
        self,
        question: str,
        chat_history: list | None,
        meeting_context: dict | None,
        base_url: str,
        api_key: str,
        model: str,
        output_language: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """UI-safe wrapper for ``query_meeting`` that surfaces friendly inline feedback."""
        normalized_history = self._normalize_chat_history(chat_history)
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            return "", normalized_history

        if not self._is_valid_meeting_context(meeting_context):
            msg = "Please generate meeting notes first before asking questions about the meeting."
            updated_history = [
                *normalized_history,
                {"role": "user", "content": cleaned_question},
                {"role": "assistant", "content": msg},
            ]
            return msg, updated_history

        try:
            return self.query_meeting(
                cleaned_question,
                normalized_history,
                meeting_context,
                base_url,
                api_key,
                model,
                output_language=output_language,
                reasoning_effort=reasoning_effort,
            )
        except gr.Error as e:
            msg = str(e)
        except Exception as e:
            msg = (
                "Unable to answer the question right now. Please check your LLM settings and try again.\n\n"
                f"Error: {type(e).__name__}: {e}"
            )

        updated_history = [
            *normalized_history,
            {"role": "user", "content": cleaned_question},
            {"role": "assistant", "content": msg},
        ]
        return msg, updated_history
