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
import sys
from datetime import datetime
from pathlib import Path

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
