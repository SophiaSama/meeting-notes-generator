"""Pure presentation view helpers and UI formatting functions for Meeting Notes Helper.

This module contains pure formatting functions that return HTML or tuples.
All functions are 100% unit-testable without running Gradio.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Literal

STARTER_QUESTIONS: list[str] = [
    "What were the action items?",
    "Who owns what?",
]


def render_provider_chip(is_configured: bool, model: str = "") -> str:
    """Render a compact status badge for the configured Provider.

    Args:
        is_configured: True if the provider API key is set/verified.
        model: Optional model identifier (e.g. 'gpt-4o-mini').

    Returns:
        HTML string with semantic CSS classes for light and dark mode styling.
    """
    clean_model = (model or "").strip()
    if is_configured:
        label = f"Connected: {html.escape(clean_model)}" if clean_model else "Connected"
        state_class = "connected"
    else:
        label = "Not configured"
        state_class = "unconfigured"

    return (
        f'<div class="provider-chip {state_class}" role="status" aria-live="polite">'
        f'<span class="provider-chip-dot" aria-hidden="true"></span>'
        f'<span class="provider-chip-text">{label}</span>'
        f"</div>"
    )


def render_empty_state() -> str:
    """Render friendly empty state placeholder for the results pane."""
    return (
        '<div class="empty-state-card" role="status">'
        '<div class="empty-state-icon" aria-hidden="true">📝</div>'
        '<div class="empty-state-title">No Meeting Notes yet</div>'
        '<div class="empty-state-hint">Add a Recording or Transcript to begin</div>'
        "</div>"
    )


def map_progress_to_stage(
    fraction: float | None,
    desc: str,
    is_transcript_only: bool = False,
) -> str:
    """Map pipeline progress fraction and description to canonical stage names.

    Stages:
        - "Extracting Screenshots"
        - "Transcribing"
        - "Summarizing"
        - "Done"
    """
    desc_lower = (desc or "").lower()
    frac_val = 0.0 if fraction is None else float(fraction)

    if frac_val >= 1.0 or "done" in desc_lower:
        return "Done"

    if is_transcript_only:
        return "Summarizing"

    if any(k in desc_lower for k in ("summariz", "calling", "condensing", "html")):
        return "Summarizing"

    if "transcrib" in desc_lower and "extract" not in desc_lower:
        return "Transcribing"

    if "extract" in desc_lower and "transcrib" not in desc_lower:
        return "Extracting Screenshots"

    if frac_val < 0.35:
        return "Extracting Screenshots"
    if frac_val < 0.60:
        return "Transcribing"
    return "Summarizing"


def format_elapsed_time(elapsed_seconds: float) -> str:
    """Format elapsed duration in seconds as 'Xs' or 'Xm YYs'."""
    total_sec = max(0, int(round(elapsed_seconds)))
    mins, secs = divmod(total_sec, 60)
    return f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"


def format_stage_progress(stage: str, elapsed_seconds: float) -> str:
    """Format the active pipeline stage and elapsed duration into styled HTML."""
    time_str = format_elapsed_time(elapsed_seconds)
    clean_stage = html.escape(stage.strip())
    is_done = clean_stage.lower() == "done"

    indicator_html = (
        '<span class="stage-check" aria-hidden="true">✓</span>'
        if is_done
        else '<span class="stage-spinner" aria-hidden="true"></span>'
    )
    extra_class = " stage-done" if is_done else ""

    return (
        f'<div class="stage-progress{extra_class}" role="status" aria-live="polite">'
        f"{indicator_html}"
        f'<span class="stage-name">{clean_stage}</span>'
        f'<span class="stage-elapsed">({time_str})</span>'
        f"</div>"
    )


def classify_error(error_text: str) -> tuple[str, str]:
    """Map a raw failure message to actionable cause and fix descriptions.

    Returns:
        tuple[cause, fix]
    """
    raw = (error_text or "").strip()
    if "Error: " in raw:
        err_part = raw.split("Error: ", 1)[-1].strip()
        if err_part:
            raw = err_part
    lower = raw.lower()

    if not raw:
        return (
            "An unexpected error occurred during processing.",
            "Check application logs and verify provider connection in Provider Settings.",
        )

    if any(k in lower for k in ("api key", "apikey", "unauthorized", "401", "authentication", "invalid_api_key", "bearer", "forbidden", "403")):
        return (
            "Invalid or missing API key.",
            "Check your API key in Provider Settings and verify account permissions or credits.",
        )

    if any(k in lower for k in ("ffmpeg", "ffprobe", "executable not found", "codec", "filenotfound")):
        return (
            "FFmpeg is not installed or not found on system PATH.",
            "Install FFmpeg and ensure it is available in your system environment PATH.",
        )

    if any(k in lower for k in ("payload", "file too large", "too large", "413", "max_tokens", "context_length", "context length", "request entity too large", "exceeds limit")):
        return (
            "Recording, Transcript, or payload exceeds model context limits.",
            "Increase Screenshot interval, reduce max Screenshots, or provide a shorter Recording or Transcript.",
        )

    if any(k in lower for k in ("upload a video", "upload a recording", "transcript file first")):
        return (
            "Missing Recording or Transcript.",
            "Add a Recording or Transcript before generating Meeting Notes.",
        )

    clean_cause = re.sub(r"\s+", " ", raw)[:240]
    return (
        f"Processing failed: {clean_cause}",
        "Check application logs, verify endpoint in Provider Settings, and try again.",
    )


def render_error_banner(cause: str, fix: str) -> str:
    """Render an inline dismissible HTML error banner."""
    esc_cause = html.escape(cause)
    esc_fix = html.escape(fix)
    return (
        '<div class="error-banner" role="alert">'
        '<div class="error-banner-icon" aria-hidden="true">⚠️</div>'
        '<div class="error-banner-body">'
        f'<div class="error-banner-cause"><strong>Error:</strong> {esc_cause}</div>'
        f'<div class="error-banner-fix"><strong>Suggested fix:</strong> {esc_fix}</div>'
        "</div>"
        '<button type="button" class="error-banner-dismiss" '
        'onclick="this.closest(\'.error-banner\').style.display=\'none\'" '
        'aria-label="Dismiss error">&times;</button>'
        "</div>"
    )


@dataclass(frozen=True)
class ResultsPaneState:
    """Observable state of the three-state Results Pane (empty, generating, populated, error)."""

    phase: Literal["empty", "generating", "populated", "error"]
    thumbnails: list[Any]
    notes_text: str
    html_notes: str
    out_path: str | None
    meeting_context: Any
    empty_visible: bool
    results_visible: bool
    error_html: str
    error_visible: bool
    starter_visible: bool
    stage_html: str
    stage_visible: bool


def begin_generation(is_transcript_only: bool = False) -> ResultsPaneState:
    """Transition the Results Pane into the generating state."""
    initial_stage = "Summarizing" if is_transcript_only else "Extracting Screenshots"
    return ResultsPaneState(
        phase="generating",
        thumbnails=[],
        notes_text="",
        html_notes="",
        out_path=None,
        meeting_context=None,
        empty_visible=False,
        results_visible=False,
        error_html="",
        error_visible=False,
        starter_visible=False,
        stage_html=format_stage_progress(initial_stage, 0.0),
        stage_visible=True,
    )


def complete_generation(
    thumbnails: list[Any],
    notes_text: str,
    html_notes: str,
    out_path: str | None,
    meeting_context: Any,
) -> ResultsPaneState:
    """Transition the Results Pane into the populated state with generated Meeting Notes."""
    return ResultsPaneState(
        phase="populated",
        thumbnails=thumbnails,
        notes_text=notes_text,
        html_notes=html_notes,
        out_path=out_path,
        meeting_context=meeting_context,
        empty_visible=False,
        results_visible=True,
        error_html="",
        error_visible=False,
        starter_visible=True,
        stage_html="",
        stage_visible=False,
    )


def fail_generation(error_text: str) -> ResultsPaneState:
    """Transition the Results Pane into the error state with an actionable banner."""
    cause, fix = classify_error(error_text)
    return ResultsPaneState(
        phase="error",
        thumbnails=[],
        notes_text="",
        html_notes="",
        out_path=None,
        meeting_context=None,
        empty_visible=True,
        results_visible=False,
        error_html=render_error_banner(cause, fix),
        error_visible=True,
        starter_visible=False,
        stage_html="",
        stage_visible=False,
    )


def reset_results_pane() -> ResultsPaneState:
    """Restore the Results Pane to its initial empty state."""
    return ResultsPaneState(
        phase="empty",
        thumbnails=[],
        notes_text="",
        html_notes="",
        out_path=None,
        meeting_context=None,
        empty_visible=True,
        results_visible=False,
        error_html="",
        error_visible=False,
        starter_visible=False,
        stage_html="",
        stage_visible=False,
    )
