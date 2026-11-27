"""
MCP Server for Meeting Notes Helper.

Exposes the meeting-notes pipeline and artifact browsing as MCP tools and
resources so that any MCP-compatible client
(e.g. Claude Desktop, VS Code Copilot) can drive the workflow
programmatically.

Run with:
    python mcp_server.py            # stdio transport (default for MCP clients)
    python mcp_server.py --sse      # SSE transport on http://localhost:8000/sse
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Bootstrap the runtime (loads .env, sets proxy env vars, etc.)
# ---------------------------------------------------------------------------
from meeting_notes.bootstrap import setup_runtime_environment, ProxyDefaults

setup_runtime_environment()

# ---------------------------------------------------------------------------
# Instantiate shared services (same wiring as the Gradio app)
# ---------------------------------------------------------------------------
from meeting_notes.llm import LLMService
from meeting_notes.video import VideoService
from meeting_notes.notes import NotesService
from meeting_notes.pipeline import PipelineService

_llm = LLMService()
_video = VideoService(_llm)
_notes = NotesService()
_pipeline = PipelineService(_llm, _video, _notes)

# ---------------------------------------------------------------------------
# Create the MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "meeting-notes-helper",
    instructions=(
        "Turn meeting recordings or transcripts into structured notes "
        "and HTML reports."
    ),
)


def _minutes_dir() -> Path:
    return Path.cwd() / "artifacts" / "meeting_minutes"


def _latest_minutes_path() -> Path | None:
    minutes_dir = _minutes_dir()
    if not minutes_dir.exists():
        return None
    candidates = [p for p in minutes_dir.glob("*.minutes.md") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))

# ---------------------------------------------------------------------------
# Helper: resolve LLM connection params from args or env defaults
# ---------------------------------------------------------------------------

def _resolve_llm_params(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Fill in blanks from environment / .env defaults."""
    defaults = ProxyDefaults.from_env()
    return {
        "base_url": (base_url or "").strip() or defaults.base_url,
        "api_key": (api_key or "").strip() or defaults.api_key,
        "model": (model or "").strip() or defaults.model,
    }


# ===================================================================
# TOOLS
# ===================================================================


@mcp.tool()
def test_llm_connection(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Test the LLM provider connection.

    All parameters are optional — values fall back to .env / environment
    defaults when omitted. `reasoning_effort` (none/minimal/low/medium/high)
    sends {"reasoning": {"effort": ...}}; it is dropped silently if rejected.
    """
    p = _resolve_llm_params(base_url, api_key, model)
    if reasoning_effort is not None:
        p["reasoning_effort"] = reasoning_effort
    return _llm.test_connection(**p)


@mcp.tool()
def generate_meeting_notes(
    video_path: str | None = None,
    transcript_path: str | None = None,
    notes_style: str = "Executive Summary",
    output_language: str = "English",
    interval_seconds: int = 30,
    max_frames: int = 8,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Generate structured meeting notes from a video recording or transcript file.

    Provide either `video_path` (absolute path to a local video file) or
    `transcript_path` (absolute path to a plain-text transcript).  When a
    video is supplied the pipeline extracts screenshots, transcribes audio,
    and generates multimodal notes.  When only a transcript is supplied,
    video processing is skipped.

    Available `notes_style` values:
      - "Executive Summary" (default)
      - "Detailed Notes"
      - "Action Items Focus"
      - "Technical Minutes"

    Optional `reasoning_effort`: "none", "minimal", "low", "medium", "high".
    When set (other than "none"), {"reasoning": {"effort": ...}} is sent to the
    model and silently dropped if the model rejects it. Omit to use the
    environment default (LLM_REASONING_EFFORT / LLM_REASONING_CONTROL_EXTRA_BODY).

    Returns a JSON object with keys: markdown_notes, html_path, minutes_path,
    screenshots_count, and artifacts_summary.
    """
    if not video_path and not transcript_path:
        return "Error: provide either video_path or transcript_path."

    p = _resolve_llm_params(base_url, api_key, model)

    # The pipeline expects Gradio-style file inputs.  For MCP we pass plain
    # path strings which PipelineService._resolve_input_*_path already handles.
    try:
        import gradio as gr

        # Provide a no-op Gradio Progress so the pipeline's progress() calls
        # are silently ignored without import or type errors.
        progress = gr.Progress(track_tqdm=False)

        thumbnails, markdown_notes, html_notes, html_path = _pipeline.run_pipeline(
            video_file=video_path,
            base_url=p["base_url"],
            api_key=p["api_key"],
            model=p["model"],
            interval_seconds=interval_seconds,
            max_frames=max_frames,
            notes_style=notes_style,
            output_language=output_language,
            transcript_file=transcript_path,
            reasoning_effort=reasoning_effort or _default_reasoning_effort(),
            progress=progress,
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    # Find the minutes file that was just written
    latest = _latest_minutes_path()
    minutes_path_str = str(latest) if latest else ""

    result = {
        "markdown_notes": markdown_notes,
        "html_path": html_path,
        "minutes_path": minutes_path_str,
        "screenshots_count": len(thumbnails) if thumbnails else 0,
    }
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def list_meeting_minutes() -> str:
    """List all saved meeting-minutes files under artifacts/meeting_minutes/.

    Returns a JSON array of objects with name, path, size_kb, and modified_at.
    """
    minutes_dir = _minutes_dir()
    if not minutes_dir.exists():
        return json.dumps([])

    entries = []
    for p in sorted(minutes_dir.glob("*.minutes.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = p.stat()
        entries.append({
            "name": p.name,
            "path": str(p),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return json.dumps(entries, indent=2)


@mcp.tool()
def get_meeting_minutes(minutes_file: str) -> str:
    """Read and return the full content of a meeting-minutes markdown file.

    `minutes_file` can be an absolute path or just the filename (resolved
    relative to artifacts/meeting_minutes/).
    """
    path = Path(minutes_file)
    if not path.is_absolute():
        path = _minutes_dir() / path
    if not path.exists():
        return f"Error: file not found: {path}"
    return path.read_text(encoding="utf-8")


@mcp.tool()
def get_latest_meeting_minutes() -> str:
    """Return the content of the most recently generated meeting-minutes file."""
    latest = _latest_minutes_path()
    if latest is None:
        return "No meeting minutes found."
    return latest.read_text(encoding="utf-8")


@mcp.tool()
def list_note_styles() -> str:
    """Return the available note-generation styles and their descriptions."""
    return json.dumps(_notes.STYLE_GUIDES, indent=2)


@mcp.tool()
def get_video_info(video_path: str) -> str:
    """Return metadata (duration, resolution, size) for a local video file."""
    try:
        info = _video.get_video_info(video_path)
        return json.dumps(info, indent=2, default=str)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _default_reasoning_effort() -> str | None:
    """Env default for MCP calls: only applied when LLM_REASONING_EFFORT is set."""
    if not os.getenv("LLM_REASONING_EFFORT", "").strip():
        return None
    return ProxyDefaults.from_env().reasoning_effort


# ---------------------------------------------------------------------------
# Helpers: Obsidian vault export & .minutes.md artifact parsing
# ---------------------------------------------------------------------------

_KNOWN_MINUTES_METADATA_KEYS = frozenset({
    "source_video",
    "source_transcript",
    "transcript_file",
    "transcript_manifest",
    "ocr_available",
    "screenshot_interval_seconds",
    "max_screenshots",
    "screenshots_captured",
    "generated_at",
    "generated",
    "source",
    "model",
    "language",
})
_MINUTES_METADATA_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*(.*)$")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_FENCE_OPEN_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_INTERNAL_BODY_SCREENSHOT_RE = re.compile(
    r"!\[[^\]]*\]\((?:\./)?(?:artifacts/meeting_minutes/)?images/([^)\n]+?)/screenshot_\d+\.(?:jpg|jpeg|png)\)",
    flags=re.IGNORECASE,
)
_SCREENSHOT_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def _parse_minutes_document(raw_text: str) -> tuple[str, dict[str, str]]:
    """Strip internal `.minutes.md` header metadata and trailing alignment checks.

    Only removes the `# Meeting Minutes` heading and its contiguous `key: value`
    metadata block when `# Meeting Minutes` is the first non-blank line, and only
    removes `## Screenshot Alignment Checks` when it appears as a trailing
    section outside fenced code blocks.
    """
    normalized = (
        (raw_text or "")
        .lstrip("\ufeff")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    lines = normalized.split("\n")

    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    metadata: dict[str, str] = {}
    if idx < len(lines) and lines[idx].strip() == "# Meeting Minutes":
        idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                break
            match = _MINUTES_METADATA_LINE_RE.match(stripped)
            if not match:
                break
            key = match.group(1)
            if (
                key.lower() not in _KNOWN_MINUTES_METADATA_KEYS
                and not re.fullmatch(r"[a-z]+_[a-z0-9_]+", key)
            ):
                break
            metadata[key.lower()] = match.group(2).strip()
            idx += 1

        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        remaining_lines = lines[idx:]
    else:
        remaining_lines = lines

    fence_char = ""
    fence_len = 0
    alignment_idx: int | None = None

    for line_idx, line in enumerate(remaining_lines):
        stripped = line.strip()
        if not fence_char:
            fence_match = _FENCE_OPEN_RE.match(line)
            if fence_match:
                fence_char = fence_match.group(1)[0]
                fence_len = len(fence_match.group(1))
                continue
        else:
            if stripped and set(stripped) == {fence_char} and len(stripped) >= fence_len:
                fence_char = ""
                fence_len = 0
            continue

        heading_match = _MARKDOWN_HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).rstrip("#").strip().lower()
            if level == 2 and heading_text == "screenshot alignment checks":
                alignment_idx = line_idx
            elif alignment_idx is not None and level <= 2:
                alignment_idx = None

    if alignment_idx is not None:
        remaining_lines = remaining_lines[:alignment_idx]

    clean_body = "\n".join(remaining_lines).strip()
    return clean_body, metadata


def _extract_clean_minutes_body(raw_text: str) -> str:
    """Return clean Markdown Meeting Notes extracted from a `.minutes.md` document."""
    clean_body, _ = _parse_minutes_document(raw_text)
    return clean_body


def _strip_minutes_suffix(name: str) -> str:
    """Strip `.minutes.md` (case-insensitive) or standard file suffix from a filename."""
    cleaned = name.strip()
    if cleaned.lower().endswith(".minutes.md"):
        return cleaned[: -len(".minutes.md")].strip()
    return Path(cleaned).stem.strip()


def _resolve_minutes_file_path(minutes_file: str) -> Path:
    """Resolve `minutes_file` as absolute, under `_minutes_dir()`, or cwd-relative."""
    raw = Path(minutes_file.strip()).expanduser()
    if raw.is_absolute():
        return raw

    in_minutes_dir = _minutes_dir() / raw
    if in_minutes_dir.exists():
        return in_minutes_dir
    if raw.exists():
        return raw
    return in_minutes_dir


def _metadata_source_stem(metadata: dict[str, str]) -> str:
    """Extract stem from `source_video`, `source_transcript`, or `source` metadata."""
    for key in ("source_video", "source_transcript", "source"):
        val = (metadata.get(key) or "").strip()
        if val:
            name = Path(val).name
            stem = name.rsplit(".", 1)[0].strip() if "." in name else name.strip()
            if stem:
                return stem
    return ""


def _extract_minutes_source_stem(minutes_path: Path, metadata: dict[str, str]) -> str:
    """Derive source stem from `.minutes.md` metadata or filename (stripping `.minutes.md`)."""
    meta_stem = _metadata_source_stem(metadata)
    if meta_stem:
        return meta_stem
    file_stem = _strip_minutes_suffix(minutes_path.name)
    return file_stem or "meeting"


def _resolve_minutes_screenshots_dir(
    minutes_path: Path,
    body: str,
    metadata: dict[str, str],
    source_stem: str,
) -> str | None:
    """Auto-discover the `images/<stem>` Screenshot directory for a `.minutes.md` file."""
    body_stems: list[str] = []
    for match in _INTERNAL_BODY_SCREENSHOT_RE.finditer(body or ""):
        decoded = unquote(match.group(1)).strip()
        if (
            not decoded
            or decoded in {".", ".."}
            or "\x00" in decoded
            or "/" in decoded
            or "\\" in decoded
            or ".." in decoded
        ):
            continue
        if decoded not in body_stems:
            body_stems.append(decoded)

    # Transcript-only artifacts without source_video and without Screenshot links have no images dir.
    if (
        metadata.get("source_transcript")
        and not metadata.get("source_video")
        and not body_stems
    ):
        return None

    candidate_stems: list[str] = []
    for stem in [
        *body_stems,
        _metadata_source_stem(metadata),
        _strip_minutes_suffix(minutes_path.name),
        source_stem.strip(),
    ]:
        if stem and stem not in candidate_stems:
            candidate_stems.append(stem)

    base_dirs: list[Path] = []
    for base in (minutes_path.parent / "images", _minutes_dir() / "images"):
        if base not in base_dirs:
            base_dirs.append(base)

    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue
        base_resolved = base_dir.resolve()
        for stem in candidate_stems:
            candidate = base_dir / stem
            if not candidate.is_dir():
                continue
            try:
                if candidate.resolve().parent != base_resolved:
                    continue
                has_screenshots = any(
                    item.is_file() and item.suffix.lower() in _SCREENSHOT_IMAGE_EXTENSIONS
                    for item in candidate.iterdir()
                )
            except OSError:
                continue
            if has_screenshots:
                return str(candidate)

    return None


def _resolve_obsidian_tool_inputs(
    *,
    minutes_file: str | None,
    notes_markdown: str | None,
    screenshots_dir: str | None,
    source_stem: str | None,
) -> tuple[str | None, str, str | None, str | None]:
    """Resolve `(error_msg, notes_body, effective_stem, effective_screenshots_dir)`."""
    has_minutes = minutes_file is not None and bool(str(minutes_file).strip())
    has_notes = notes_markdown is not None and bool(str(notes_markdown).strip())

    if not has_minutes and not has_notes:
        return "Error: provide either minutes_file or notes_markdown.", "", None, None

    explicit_stem = (source_stem or "").strip() or None
    explicit_shots = (
        str(screenshots_dir).strip()
        if screenshots_dir is not None and str(screenshots_dir).strip()
        else None
    )

    if has_minutes:
        minutes_path = _resolve_minutes_file_path(str(minutes_file))
        if not minutes_path.exists():
            return f"Error: file not found: {minutes_path}", "", None, None
        if not minutes_path.is_file():
            return f"Error: minutes_file is not a file: {minutes_path}", "", None, None
        try:
            raw_text = minutes_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}", "", None, None

        clean_body, metadata = _parse_minutes_document(raw_text)
        effective_stem = explicit_stem or _extract_minutes_source_stem(minutes_path, metadata)
        resolved_notes = str(notes_markdown) if has_notes else clean_body
        if not resolved_notes.strip():
            return "Error: no Meeting Notes content found in minutes_file.", "", None, None

        effective_shots = (
            explicit_shots
            if explicit_shots is not None
            else _resolve_minutes_screenshots_dir(
                minutes_path, resolved_notes, metadata, effective_stem
            )
        )
        return None, resolved_notes, effective_stem, effective_shots

    return None, str(notes_markdown), explicit_stem, explicit_shots


@mcp.tool()
def prepare_obsidian_export(
    vault_path: str,
    minutes_file: str | None = None,
    notes_markdown: str | None = None,
    screenshots_dir: str | None = None,
    source_stem: str | None = None,
    chat_history: list | str | None = None,
    attachment_prefix: str | None = None,
    output_language: str | None = None,
    export_date: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Preview an Obsidian vault export without writing any files to disk.

    Validates `vault_path`, extracts clean Meeting Notes from `minutes_file`
    (stripping internal metadata headers and trailing Screenshot alignment
    checks) or uses `notes_markdown`, derives the `YYYY-MM-DD <Title>` target
    folder, checks for existing folder collisions (`exists`), counts available
    Screenshot files in `screenshots_dir`, and distills optional `Meeting Chat`
    (`chat_history`) into a FAQ draft via the configured `Provider`.

    `minutes_file` may be an absolute path or a filename under
    `artifacts/meeting_minutes/`. Returns a JSON object string, or an `"Error: ..."`
    message on validation or `Provider` failure.
    """
    error_msg, resolved_notes, effective_stem, effective_shots = _resolve_obsidian_tool_inputs(
        minutes_file=minutes_file,
        notes_markdown=notes_markdown,
        screenshots_dir=screenshots_dir,
        source_stem=source_stem,
    )
    if error_msg:
        return error_msg

    p = _resolve_llm_params(base_url, api_key, model)
    effective_effort = (
        reasoning_effort if reasoning_effort is not None else _default_reasoning_effort()
    )

    try:
        prepared = _pipeline.prepare_obsidian_export(
            vault_path=vault_path,
            notes_markdown=resolved_notes,
            source_stem=effective_stem,
            screenshots_dir=effective_shots,
            chat_history=chat_history,
            base_url=p["base_url"],
            api_key=p["api_key"],
            model=p["model"],
            output_language=output_language,
            reasoning_effort=effective_effort,
            export_date=export_date,
            attachment_prefix=attachment_prefix,
        )
        return json.dumps(prepared, indent=2, default=str)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
def export_to_obsidian(
    vault_path: str,
    minutes_file: str | None = None,
    notes_markdown: str | None = None,
    screenshots_dir: str | None = None,
    source_stem: str | None = None,
    chat_history: list | str | None = None,
    faq_markdown: str | None = None,
    attachment_prefix: str | None = None,
    overwrite: bool = False,
    output_language: str | None = None,
    export_date: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Statelessly export Meeting Notes, Screenshot attachments, and optional FAQ to an Obsidian vault.

    Accepts either `minutes_file` (saved `.minutes.md` artifact, with internal
    headers and trailing Screenshot alignment checks stripped automatically) or
    explicit `notes_markdown` (plus optional `screenshots_dir`). Rewrites
    Screenshot links into `attachment/<prefix>_NN.<ext>` and copies Screenshot
    files non-destructively (omitting `attachment/` when zero Screenshots exist).

    When `faq_markdown` is provided (including `""`), it overrides `chat_history`
    and skips `Provider` FAQ distillation (`""` suppresses the FAQ file). When
    `faq_markdown` is omitted and `chat_history` is provided, the `Provider`
    distills `Meeting Chat` into `<YYYY-MM-DD Title> - FAQ.md`.

    Collision protection: when `overwrite=False` (default) and the target
    meeting folder already exists, returns a JSON conflict payload (`status="conflict"`,
    `written=False`, `exported=False`, `error=...`) without modifying any files.
    When `overwrite=True`, selectively replaces exported files while preserving
    unrelated user files.
    """
    error_msg, resolved_notes, effective_stem, effective_shots = _resolve_obsidian_tool_inputs(
        minutes_file=minutes_file,
        notes_markdown=notes_markdown,
        screenshots_dir=screenshots_dir,
        source_stem=source_stem,
    )
    if error_msg:
        return error_msg

    p = _resolve_llm_params(base_url, api_key, model)
    effective_effort = (
        reasoning_effort if reasoning_effort is not None else _default_reasoning_effort()
    )
    effective_chat_history = None if faq_markdown is not None else chat_history

    try:
        prepared = _pipeline.prepare_obsidian_export(
            vault_path=vault_path,
            notes_markdown=resolved_notes,
            source_stem=effective_stem,
            screenshots_dir=effective_shots,
            chat_history=effective_chat_history,
            base_url=p["base_url"],
            api_key=p["api_key"],
            model=p["model"],
            output_language=output_language,
            reasoning_effort=effective_effort,
            export_date=export_date,
            attachment_prefix=attachment_prefix,
        )
        export_kwargs: dict[str, Any] = {
            "attachment_prefix": attachment_prefix,
            "overwrite": overwrite,
        }
        if faq_markdown is not None:
            export_kwargs["faq_markdown"] = faq_markdown

        result = _pipeline.export_to_obsidian(prepared, **export_kwargs)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ===================================================================
# RESOURCES
# ===================================================================


@mcp.resource("meeting-notes://minutes/latest")
def resource_latest_minutes() -> str:
    """The most recently generated meeting-minutes markdown content."""
    latest = _latest_minutes_path()
    if latest is None:
        return "No meeting minutes found yet."
    return latest.read_text(encoding="utf-8")


@mcp.resource("meeting-notes://minutes/list")
def resource_minutes_list() -> str:
    """JSON list of all saved meeting-minutes files."""
    return list_meeting_minutes()


@mcp.resource("meeting-notes://config/status")
def resource_config_status() -> str:
    """Current configuration status showing loaded .env defaults."""
    defaults = ProxyDefaults.from_env()
    return defaults.status_markdown()


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"
    mcp.run(transport=transport)
