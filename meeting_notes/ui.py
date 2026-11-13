import os
import socket
import time
from typing import Any, Callable

from .bootstrap import ProxyDefaults, setup_runtime_environment

setup_runtime_environment()

import gradio as gr

from .llm import REASONING_EFFORT_CHOICES, LLMService
from .notes import NotesService
from .pipeline import PipelineService
from .video import VideoService
from .views import (
    STARTER_QUESTIONS,
    ResultsPaneState,
    begin_generation,
    complete_generation,
    fail_generation,
    format_elapsed_time,
    format_stage_progress,
    map_progress_to_stage,
    render_empty_state,
    render_provider_chip,
    reset_results_pane,
)

THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
)

CSS = """
html body {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    color: var(--body-text-color) !important;
}
.gradio-container { max-width: 100% !important; width: 100% !important; padding-left: 24px !important; padding-right: 24px !important; }
.gradio-container > .main, .gradio-container .main .wrap, .gradio-container .contain { max-width: 100% !important; }
.title-row {
    margin-bottom: .25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.theme-toggle-btn {
    width: fit-content !important;
    flex: 0 0 auto !important;
    white-space: nowrap !important;
}
.chat-input-row {
    align-items: stretch !important;
    gap: 8px !important;
}
.subtitle   { font-size: 13px; color: var(--body-text-color-subdued); margin-top: 2px; }
.note-box   { font-family: monospace; font-size: 12px; }
.capture-hint-box {
    padding: 8px 10px;
    border: 1px solid var(--color-accent);
    background: var(--background-fill-secondary);
    border-radius: 6px;
    color: var(--body-text-color);
    font-size: 13px;
    line-height: 1.45;
}
.capture-hint-box strong {
    color: var(--body-text-color);
    font-weight: 700;
}
@media (max-width: 900px) {
    .gradio-container { padding-left: 12px !important; padding-right: 12px !important; }
    .main-row { flex-direction: column !important; }
    .main-row > .column { min-width: 100% !important; width: 100% !important; }
    .generate-btn {
        position: sticky;
        bottom: 8px;
        z-index: 20;
        box-shadow: var(--shadow-drop-lg);
    }
}
/* Provider Status Chip */
.provider-chip-container {
    margin-bottom: 8px;
}
.provider-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 500;
    line-height: 1.4;
    transition: all 0.2s ease-in-out;
}
.provider-chip.connected {
    background: rgba(16, 185, 129, 0.12);
    color: #065f46;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
:root.dark .provider-chip.connected, .dark .provider-chip.connected {
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    border-color: rgba(16, 185, 129, 0.4);
}
.provider-chip.connected .provider-chip-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
}
.provider-chip.unconfigured {
    background: var(--background-fill-secondary);
    color: var(--body-text-color-subdued);
    border: 1px solid var(--border-color-primary, #cbd5e1);
}
:root.dark .provider-chip.unconfigured, .dark .provider-chip.unconfigured {
    background: var(--background-fill-secondary);
    color: var(--body-text-color-subdued);
    border-color: var(--border-color-primary, #475569);
}
.provider-chip.unconfigured .provider-chip-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #94a3b8;
}

/* Accordion Highlight for Unconfigured Provider */
.provider-accordion-unconfigured {
    border: 2px solid var(--color-accent) !important;
    border-radius: var(--block-radius, 8px);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

/* Results Pane Empty State */
.results-empty-state-wrapper {
    width: 100%;
}
.empty-state-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    border: 1.5px dashed var(--border-color-primary, #cbd5e1);
    border-radius: var(--block-radius, 8px);
    background: var(--background-fill-secondary);
    text-align: center;
    min-height: 320px;
    margin-top: 4px;
}
:root.dark .empty-state-card, .dark .empty-state-card {
    border-color: var(--border-color-primary, #475569);
    background: var(--background-fill-secondary);
}
.empty-state-icon {
    font-size: 40px;
    margin-bottom: 12px;
    line-height: 1;
}
.empty-state-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--body-text-color);
    margin-bottom: 6px;
}
.empty-state-hint {
    font-size: 13px;
    color: var(--body-text-color-subdued);
}

/* Quick Share Row in Rendered Notes */
.share-actions-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    justify-content: flex-end;
}
.results-container {
    width: 100%;
}
.rendered-notes-html {
    background: var(--background-fill-primary) !important;
    color: var(--body-text-color) !important;
    padding: 20px !important;
    border-radius: 8px;
    border: 1px solid var(--border-color-primary, #cbd5e1);
    max-height: 68vh;
    overflow-y: auto;
}
.rendered-notes-html,
.rendered-notes-html h1,
.rendered-notes-html h2,
.rendered-notes-html h3,
.rendered-notes-html h4,
.rendered-notes-html p,
.rendered-notes-html li,
.rendered-notes-html td,
.rendered-notes-html th,
.rendered-notes-html strong,
.rendered-notes-html code {
    color: var(--body-text-color) !important;
}
.rendered-notes-html .meta,
.rendered-notes-html figcaption {
    color: var(--body-text-color-subdued) !important;
}
.rendered-notes-html blockquote,
.rendered-notes-html figcaption {
    background: var(--background-fill-secondary) !important;
    color: var(--body-text-color-subdued) !important;
}
/* Stage Progress Indicator */
.stage-progress-container {
    margin-bottom: 8px;
}
.stage-progress {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 500;
    line-height: 1.4;
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary, #cbd5e1);
    margin-bottom: 8px;
}
.stage-progress.stage-done {
    background: rgba(16, 185, 129, 0.12);
    color: #065f46;
    border-color: rgba(16, 185, 129, 0.3);
}
:root.dark .stage-progress.stage-done, .dark .stage-progress.stage-done {
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    border-color: rgba(16, 185, 129, 0.4);
}
.stage-spinner {
    width: 12px;
    height: 12px;
    border: 2px solid var(--color-accent);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.stage-check {
    font-weight: bold;
    color: #10b981;
}
.stage-elapsed {
    color: var(--body-text-color-subdued);
    font-size: 12px;
}

/* Actionable Error Banner */
.error-banner-container {
    width: 100%;
    margin-bottom: 8px;
}
.error-banner {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    border-radius: var(--block-radius, 8px);
    background: #fef2f2 !important;
    border: 1px solid #f87171 !important;
    color: #7f1d1d !important;
    box-sizing: border-box;
    width: 100%;
    margin-bottom: 12px;
    font-size: 13px;
    line-height: 1.45;
    position: relative;
    transition: opacity 0.2s ease-in-out;
}
.error-banner * {
    color: #7f1d1d !important;
}
:root.dark .error-banner, .dark .error-banner {
    background: #3b1219 !important;
    border-color: #f87171 !important;
    color: #fecaca !important;
}
:root.dark .error-banner *, .dark .error-banner * {
    color: #fecaca !important;
}
.error-banner-icon {
    font-size: 18px;
    line-height: 1.2;
    flex-shrink: 0;
}
.error-banner-body {
    flex: 1;
    font-size: 13px;
    line-height: 1.5;
}
.error-banner-cause {
    margin-bottom: 4px;
    color: inherit;
}
.error-banner-cause strong, .error-banner-fix strong {
    font-weight: 700;
}
.error-banner-fix {
    color: inherit;
    font-size: 12.5px;
    opacity: 0.95;
}
.error-banner-dismiss {
    background: transparent;
    border: none;
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    color: inherit;
    opacity: 0.7;
    padding: 2px 6px;
    border-radius: 4px;
    transition: opacity 0.15s ease, background-color 0.15s ease;
}
.error-banner-dismiss:hover {
    opacity: 1;
    background-color: rgba(0, 0, 0, 0.08);
}
:root.dark .error-banner-dismiss:hover, .dark .error-banner-dismiss:hover {
    background-color: rgba(255, 255, 255, 0.12);
}
.error-banner-dismiss:focus-visible {
    outline: 2px solid var(--color-accent);
}

/* Meeting Chat Starter Question Chips */
.chat-starters-row, .starter-chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 6px;
    margin-bottom: 8px;
    align-items: center;
}
.chat-starter-btn, .starter-chip-btn {
    border-radius: 9999px !important;
    padding: 6px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.35 !important;
    white-space: nowrap !important;
    width: fit-content !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
    background: var(--background-fill-secondary) !important;
    color: var(--body-text-color) !important;
    border: 1px solid var(--border-color-primary, #cbd5e1) !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
}
.chat-starter-btn:hover, .starter-chip-btn:hover {
    background: var(--background-fill-primary) !important;
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
}
:root.dark .chat-starter-btn, :root.dark .starter-chip-btn,
.dark .chat-starter-btn, .dark .starter-chip-btn {
    background: var(--background-fill-secondary) !important;
    color: var(--body-text-color) !important;
    border-color: var(--border-color-primary, #475569) !important;
}
:root.dark .chat-starter-btn:hover, :root.dark .starter-chip-btn:hover,
.dark .chat-starter-btn:hover, .dark .starter-chip-btn:hover {
    border-color: var(--color-accent) !important;
    background: rgba(59, 130, 246, 0.15) !important;
    color: #93c5fd !important;
}
.chat-starter-btn:focus-visible, .starter-chip-btn:focus-visible {
    outline: 2px solid var(--color-accent) !important;
    outline-offset: 2px !important;
}
footer      { display: none !important; }
"""

TOGGLE_THEME_JS = """() => {
    const root = document.documentElement;
    const body = document.body;
    const isDark = root.classList.contains('dark') || body.classList.contains('dark');
    root.classList.toggle('dark', !isDark);
    body.classList.toggle('dark', !isDark);
}
"""

COPY_MARKDOWN_JS = """(text) => {
    if (!text) return;
    const copyText = (t) => {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(t);
        }
        const ta = document.createElement('textarea');
        ta.value = t;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        return Promise.resolve();
    };
    copyText(text).then(() => {
        const btn = document.querySelector('.copy-md-btn') || document.querySelector('.copy-markdown-btn');
        if (btn) {
            const orig = btn.innerText;
            btn.innerText = 'Copied!';
            setTimeout(() => { btn.innerText = orig; }, 2000);
        }
    }).catch(err => console.error('Copy failed:', err));
}"""


class StageProgressTracker:
    """Non-invasive progress adapter wrapping Gradio's progress callable.

    Maps raw pipeline (fraction, desc) callbacks to 4 canonical stages:
    - "Extracting Screenshots"
    - "Transcribing"
    - "Summarizing"
    - "Done"
    Tracks elapsed duration from run start and forwards formatted stage
    descriptions to Gradio's live progress indicator.
    """

    def __init__(
        self,
        progress_fn: Callable[..., Any] | None = None,
        is_transcript_only: bool = False,
    ):
        self.progress_fn = progress_fn
        self.is_transcript_only = is_transcript_only
        self.start_time = time.monotonic()
        self.current_stage: str | None = None
        self.elapsed_seconds: float = 0.0
        self.last_fraction: float | None = None
        self.last_desc: str | None = None
        self.last_html: str = ""
        self.history: list[dict[str, Any]] = []

    def __call__(
        self,
        fraction: float | None = None,
        desc: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        elapsed = max(0.0, time.monotonic() - self.start_time)
        self.elapsed_seconds = elapsed
        self.last_fraction = fraction
        raw_desc = desc if desc is not None else kwargs.pop("desc", None)
        self.last_desc = raw_desc

        frac_val = 0.0 if fraction is None else float(fraction)
        stage = map_progress_to_stage(
            frac_val,
            raw_desc or "",
            is_transcript_only=self.is_transcript_only,
        )
        self.current_stage = stage

        time_str = format_elapsed_time(elapsed)
        formatted_desc = f"{stage} ({time_str})"
        self.last_html = format_stage_progress(stage, elapsed)

        self.history.append({
            "fraction": fraction,
            "raw_desc": raw_desc,
            "stage": stage,
            "elapsed": elapsed,
            "formatted_desc": formatted_desc,
        })

        if self.progress_fn is not None and callable(self.progress_fn):
            try:
                return self.progress_fn(fraction, desc=formatted_desc, *args, **kwargs)
            except TypeError:
                return self.progress_fn(fraction, *args, **kwargs)
        return None

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookup (e.g. tqdm methods) to underlying progress object."""
        if self.progress_fn is not None:
            return getattr(self.progress_fn, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class MeetingNotesApp:
    """UI composition and app lifecycle."""

    def __init__(
        self,
        *,
        llm_service: LLMService | None = None,
        video_service: VideoService | None = None,
        notes_service: NotesService | None = None,
        pipeline_service: PipelineService | None = None,
    ):
        self.defaults = ProxyDefaults.from_env()
        if pipeline_service is not None:
            self.pipeline_service = pipeline_service
            self.llm_service = llm_service or pipeline_service.llm_service
            self.video_service = video_service or pipeline_service.video_service
            self.notes_service = notes_service or pipeline_service.notes_service
        else:
            self.llm_service = llm_service or LLMService()
            self.video_service = video_service or VideoService(self.llm_service)
            self.notes_service = notes_service or NotesService()
            self.pipeline_service = PipelineService(
                self.llm_service,
                self.video_service,
                self.notes_service,
            )
        self.demo = self._build_demo()

    def _suggest_capture_settings(self, video_path):
        """Suggest capture settings that keep the final multimodal request under budget."""
        if not video_path:
            return gr.update(), gr.update(), gr.update(value="", visible=False)
        try:
            info = self.video_service.get_video_info(video_path)
            duration = float(info["duration"])
        except Exception:
            return gr.update(), gr.update(), gr.update(value="", visible=False)
        if duration <= 0:
            return gr.update(), gr.update(), gr.update(value="", visible=False)

        avg_frame_request_bytes = self._sample_average_frame_request_bytes(video_path, duration)
        recommendation = self.pipeline_service.recommend_capture_settings(
            duration_seconds=duration,
            resolution=info.get("resolution", ""),
            avg_frame_request_bytes=avg_frame_request_bytes,
        )

        estimated_mb = recommendation["estimated_request_bytes"] / 1_048_576
        budget_mb = recommendation["budget_bytes"] / 1_048_576
        basis = "sampled screenshots" if avg_frame_request_bytes else "duration and resolution"
        hint_html = (
            "<div class='capture-hint-box'>"
            "<strong>Recommended capture settings:</strong> "
            f"every {recommendation['interval_seconds']}s, up to {recommendation['max_frames']} screenshots "
            f"for a {self.video_service.format_duration(duration)} Recording. "
            f"Estimated Meeting Notes request: {estimated_mb:.1f} MB of {budget_mb:.1f} MB "
            f"based on {basis}."
            "</div>"
        )
        return (
            gr.update(value=recommendation["interval_seconds"]),
            gr.update(value=recommendation["max_frames"]),
            gr.update(value=hint_html, visible=True),
        )

    def _sample_average_frame_request_bytes(self, video_path: str, duration: float) -> int | None:
        sample_count = 3
        try:
            sample_interval = max(10, int(duration / (sample_count + 1)))
            frames = self.video_service.extract_frames(
                video_path,
                interval_seconds=sample_interval,
                max_frames=sample_count,
            )
        except Exception:
            return None

        sizes = [len(frame.get("base64_jpeg", "").encode("utf-8")) + 256 for frame in frames]
        if not sizes:
            return None
        return max(1, round(sum(sizes) / len(sizes)))

    def _on_video_input_change(self, video_path):
        """Handle upload updates in one callback to avoid extra duplicate work."""
        duplicate_banner_update = self.pipeline_service.check_upload_for_duplicate(video_path)
        if isinstance(duplicate_banner_update, dict) and duplicate_banner_update.get("visible"):
            return gr.update(), gr.update(), gr.update(value="", visible=False), duplicate_banner_update

        interval_update, max_frames_update, capture_hint_update = self._suggest_capture_settings(video_path)
        return interval_update, max_frames_update, capture_hint_update, duplicate_banner_update

    def _on_test_connection(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Test LLM connectivity and return result text, status chip HTML, and accordion state."""
        res = self.llm_service.test_connection(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        is_connected = res.startswith("✅")
        chip_html = render_provider_chip(
            is_configured=is_connected,
            model=model if is_connected else "",
        )
        accordion_classes = ["provider-accordion"]
        if not is_connected:
            accordion_classes.append("provider-accordion-unconfigured")
        accordion_update = gr.update(
            open=not is_connected,
            elem_classes=accordion_classes,
        )
        return res, chip_html, accordion_update

    @staticmethod
    def _adapt_results_updates(state: ResultsPaneState) -> tuple[Any, Any, Any, Any, Any]:
        """Adapt ResultsPaneState into Gradio component updates for the Results Pane."""
        return (
            gr.update(visible=state.empty_visible),
            gr.update(visible=state.results_visible),
            gr.update(value=state.error_html, visible=state.error_visible),
            gr.update(visible=state.starter_visible),
            gr.update(value=state.stage_html, visible=state.stage_visible),
        )

    @classmethod
    def _on_generate_start(cls, video_file=None, transcript_file=None):
        """Switch results pane into the generating state with stage progress HTML."""
        state = begin_generation(is_transcript_only=bool(transcript_file and not video_file))
        empty_u, results_u, error_u, _starter_u, stage_u = cls._adapt_results_updates(state)
        return empty_u, results_u, error_u, stage_u

    def _on_generate_click(
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
        """Generate notes, store the session's MeetingContext, and reset prior Q&A chat."""
        is_transcript_only = bool(transcript_file and not video_file)
        if isinstance(progress, StageProgressTracker):
            tracker = progress
        else:
            tracker = StageProgressTracker(
                progress_fn=progress,
                is_transcript_only=is_transcript_only,
            )

        error_msg = None
        thumbnails, notes_text, html_notes, out_path, meeting_context = [], "", "", None, None

        try:
            thumbnails, notes_text, html_notes, out_path, meeting_context = (
                self.pipeline_service.run_pipeline_safe(
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
                    progress=tracker,
                    return_context=True,
                )
            )
            if not self.pipeline_service._is_valid_meeting_context(meeting_context):
                meeting_context = None
                error_msg = notes_text or "Generation failed without returning valid meeting context."
        except (gr.Error, Exception) as exc:
            meeting_context = None
            error_msg = str(exc)

        if error_msg or meeting_context is None:
            state = fail_generation(error_msg or "")
        else:
            state = complete_generation(
                thumbnails=thumbnails,
                notes_text=notes_text,
                html_notes=html_notes,
                out_path=out_path,
                meeting_context=meeting_context,
            )

        return (
            state.thumbnails,
            state.notes_text,
            state.html_notes,
            state.out_path,
            state.meeting_context,
            [],
            "",
            *self._adapt_results_updates(state),
        )

    def _on_chat_submit(
        self,
        question,
        chat_history,
        meeting_context,
        base_url,
        api_key,
        model,
        output_language=None,
        reasoning_effort=None,
    ):
        """Handle a Q&A turn against the active session's MeetingContext."""
        _reply, updated_history = self.pipeline_service.query_meeting_safe(
            question,
            chat_history,
            meeting_context,
            base_url,
            api_key,
            model,
            output_language=output_language,
            reasoning_effort=reasoning_effort,
        )
        return updated_history, ""

    @staticmethod
    def _on_starter_click(question: str) -> Any:
        """Prefill the question input with the selected starter prompt without sending."""
        return gr.update(value=question)

    @staticmethod
    def _on_clear_chat():
        """Clear only the Q&A conversation history and question input."""
        return [], ""

    @classmethod
    def _on_reset(cls):
        """Reset all inputs, generated outputs, session MeetingContext, and Q&A history."""
        state = reset_results_pane()
        return (
            None,
            None,
            state.thumbnails,
            state.notes_text,
            state.html_notes,
            state.out_path,
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            state.meeting_context,
            [],
            "",
            *cls._adapt_results_updates(state),
        )

    @staticmethod
    def find_available_port(start: int = 7860, end: int = 7870, host: str = "127.0.0.1") -> int:
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((host, port))
                    return port
                except OSError:
                    continue
        raise OSError(f"Cannot find empty port in range: {start}-{end}.")

    def _build_demo(self):
        with gr.Blocks(title="Meeting Notes Helper") as demo:
            meeting_context_state = gr.State(value=None)
            self.meeting_context_state = meeting_context_state

            with gr.Row(elem_classes="title-row"):
                gr.Markdown("# Meeting Notes Helper")
                theme_toggle_btn = gr.Button(
                    "Theme: Light / Dark",
                    size="sm",
                    variant="secondary",
                    scale=0,
                    elem_classes=["theme-toggle-btn"],
                )
            gr.Markdown(
                "Add a Recording or a Transcript -> generate structured Meeting Notes via any OpenAI-compatible model.",
                elem_classes="subtitle",
            )

            is_configured = bool(self.defaults.api_key and self.defaults.api_key.strip())
            provider_status_chip = gr.HTML(
                value=render_provider_chip(
                    is_configured=is_configured,
                    model=self.defaults.model,
                ),
                elem_classes=["provider-chip-container"],
            )

            accordion_classes = ["provider-accordion"]
            if not is_configured:
                accordion_classes.append("provider-accordion-unconfigured")

            with gr.Accordion(
                "Provider Settings",
                open=not is_configured,
                elem_classes=accordion_classes,
            ) as provider_accordion:
                with gr.Row():
                    base_url_input = gr.Textbox(
                        label="Base URL",
                        value=self.defaults.base_url,
                        placeholder="https://api.openai.com/v1",
                        scale=3,
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        value=self.defaults.api_key,
                        placeholder="sk-...",
                        type="password",
                        scale=2,
                    )
                    model_input = gr.Textbox(
                        label="Model",
                        value=self.defaults.model,
                        placeholder="gpt-4o-mini",
                        scale=2,
                    )
                    reasoning_effort_input = gr.Dropdown(
                        choices=list(REASONING_EFFORT_CHOICES),
                        value=self.defaults.reasoning_effort,
                        label="Reasoning effort",
                        info='Sends {"reasoning": {"effort": ...}}; dropped automatically if the model rejects it. "none" sends nothing.',
                        scale=1,
                    )
                with gr.Row():
                    test_btn = gr.Button("Test Connection", variant="secondary", scale=0)
                    test_result = gr.Textbox(label="", show_label=False, interactive=False, scale=4)

                test_btn.click(
                    self._on_test_connection,
                    inputs=[
                        base_url_input,
                        api_key_input,
                        model_input,
                        reasoning_effort_input,
                    ],
                    outputs=[
                        test_result,
                        provider_status_chip,
                        provider_accordion,
                    ],
                )

                gr.Markdown(self.defaults.status_markdown(), elem_classes="subtitle")

            with gr.Row(elem_classes="main-row"):
                with gr.Column(scale=1):
                    video_input = gr.File(
                        label="Recording",
                        file_types=["video"],
                        type="filepath",
                    )
                    transcript_input = gr.File(
                        label="Transcript (optional, skips Recording processing)",
                        file_types=[".txt", ".md", ".rtf"],
                        type="filepath",
                    )

                    with gr.Row():
                        interval_slider = gr.Slider(
                            minimum=10,
                            maximum=600,
                            value=30,
                            step=5,
                            label="Screenshot interval (seconds)",
                        )
                        max_frames_slider = gr.Slider(
                            minimum=1,
                            maximum=120,
                            value=8,
                            step=1,
                            label="Max screenshots",
                        )
                    capture_hint = gr.HTML(visible=False)
                    with gr.Row():
                        notes_style_dd = gr.Dropdown(
                            choices=list(self.notes_service.STYLE_GUIDES.keys()),
                            value="Executive Summary",
                            label="Notes Style",
                        )
                        language_dd = gr.Dropdown(
                            choices=[
                                "English",
                                "Spanish",
                                "French",
                                "German",
                                "Japanese",
                                "Chinese",
                                "Portuguese",
                                "Korean",
                            ],
                            value="English",
                            label="Meeting Notes language",
                        )

                    generate_btn = gr.Button("Generate Meeting Notes", variant="primary", elem_classes="generate-btn")
                    reset_btn = gr.Button("Reset", variant="secondary")

                with gr.Column(scale=2):
                    dup_banner = gr.HTML(visible=False)
                    error_banner = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["error-banner-container"],
                    )
                    stage_progress_html = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes=["stage-progress-container"],
                    )
                    empty_state_container = gr.HTML(
                        value=render_empty_state(),
                        visible=True,
                        elem_classes=["results-empty-state-wrapper"],
                    )
                    with gr.Column(visible="hidden", elem_classes=["results-container"]) as results_container:
                        screenshots_gallery = gr.Gallery(
                            label="Screenshots",
                            columns=4,
                            height=220,
                            show_label=True,
                        )
                        with gr.Tabs():
                            with gr.TabItem("Rendered Meeting Notes"):
                                with gr.Row(elem_classes=["share-actions-row"]):
                                    copy_md_btn = gr.Button(
                                        "Copy Markdown",
                                        elem_classes=["copy-md-btn", "copy-markdown-btn"],
                                        size="sm",
                                    )
                                    download_btn = gr.DownloadButton(
                                        "Download HTML",
                                        value=None,
                                        elem_classes=["download-btn"],
                                        size="sm",
                                    )
                                html_notes = gr.HTML(label="Rendered Meeting Notes", elem_classes=["rendered-notes-html"])
                            with gr.TabItem("Markdown"):
                                notes_text = gr.Textbox(
                                    label="",
                                    lines=20,
                                    buttons=["copy"],
                                    elem_classes="note-box",
                                )
                            with gr.TabItem("Meeting Chat"):
                                chatbot = gr.Chatbot(
                                    label="Meeting Chat",
                                    height="48vh",
                                    placeholder="Generate Meeting Notes first, then ask questions about the meeting here.",
                                )
                                starter_row = gr.Row(
                                    visible=False,
                                    elem_classes=["chat-starters-row", "starter-chips-row"],
                                )
                                starter_buttons: list[gr.Button] = []
                                with starter_row:
                                    for question in STARTER_QUESTIONS:
                                        btn = gr.Button(
                                            question,
                                            size="sm",
                                            variant="secondary",
                                            elem_classes=["chat-starter-btn", "starter-chip-btn"],
                                            scale=0,
                                        )
                                        starter_buttons.append(btn)
                                with gr.Row(elem_classes=["chat-input-row"]):
                                    chat_input = gr.Textbox(
                                        label="Question",
                                        placeholder="Ask a question about the meeting...",
                                        show_label=False,
                                        scale=6,
                                        lines=2,
                                        max_lines=4,
                                    )
                                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=90)
                                    clear_chat_btn = gr.Button("Clear Chat", variant="secondary", scale=0, min_width=105)

                                for btn, question in zip(starter_buttons, STARTER_QUESTIONS):
                                    btn.click(
                                        fn=lambda q=question: self._on_starter_click(q),
                                        inputs=[],
                                        outputs=[chat_input],
                                        show_progress="hidden",
                                    )

            self.video_input = video_input
            self.transcript_input = transcript_input
            self.screenshots_gallery = screenshots_gallery
            self.notes_text = notes_text
            self.notes_output = notes_text
            self.html_notes = html_notes
            self.html_preview = html_notes
            self.download_btn = download_btn
            self.download_file = download_btn
            self.empty_state_container = empty_state_container
            self.results_container = results_container
            self.error_banner = error_banner
            self.stage_progress_html = stage_progress_html
            self.copy_md_btn = copy_md_btn
            self.dup_banner = dup_banner
            self.capture_hint = capture_hint
            self.chatbot = chatbot
            self.chat_input = chat_input
            self.starter_row = starter_row
            self.starter_buttons = starter_buttons
            self.send_btn = send_btn
            self.clear_chat_btn = clear_chat_btn
            self.generate_btn = generate_btn
            self.reset_btn = reset_btn
            self.provider_status_chip = provider_status_chip
            self.provider_accordion = provider_accordion
            self.base_url_input = base_url_input
            self.api_key_input = api_key_input
            self.model_input = model_input
            self.reasoning_effort_input = reasoning_effort_input
            self.test_btn = test_btn
            self.test_result = test_result

            self.theme_toggle_btn = theme_toggle_btn
            theme_toggle_btn.click(
                None,
                inputs=[],
                outputs=[],
                js=TOGGLE_THEME_JS,
            )

            copy_md_btn.click(
                None,
                inputs=[notes_text],
                outputs=[],
                js=COPY_MARKDOWN_JS,
            )

            generate_btn.click(
                self._on_generate_start,
                inputs=[video_input, transcript_input],
                outputs=[
                    empty_state_container,
                    results_container,
                    error_banner,
                    stage_progress_html,
                ],
                show_progress="hidden",
            ).then(
                self._on_generate_click,
                inputs=[
                    video_input,
                    base_url_input,
                    api_key_input,
                    model_input,
                    interval_slider,
                    max_frames_slider,
                    notes_style_dd,
                    language_dd,
                    transcript_input,
                    reasoning_effort_input,
                ],
                outputs=[
                    screenshots_gallery,
                    notes_text,
                    html_notes,
                    download_btn,
                    meeting_context_state,
                    chatbot,
                    chat_input,
                    empty_state_container,
                    results_container,
                    error_banner,
                    starter_row,
                    stage_progress_html,
                ],
            )

            chat_inputs = [
                chat_input,
                chatbot,
                meeting_context_state,
                base_url_input,
                api_key_input,
                model_input,
                language_dd,
                reasoning_effort_input,
            ]
            send_btn.click(
                self._on_chat_submit,
                inputs=chat_inputs,
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                self._on_chat_submit,
                inputs=chat_inputs,
                outputs=[chatbot, chat_input],
            )
            clear_chat_btn.click(
                self._on_clear_chat,
                inputs=[],
                outputs=[chatbot, chat_input],
            )

            video_input.change(
                self._on_video_input_change,
                inputs=[video_input],
                outputs=[interval_slider, max_frames_slider, capture_hint, dup_banner],
                show_progress="hidden",
            )

            reset_btn.click(
                self._on_reset,
                inputs=[],
                outputs=[
                    video_input,
                    transcript_input,
                    screenshots_gallery,
                    notes_text,
                    html_notes,
                    download_btn,
                    dup_banner,
                    capture_hint,
                    meeting_context_state,
                    chatbot,
                    chat_input,
                    empty_state_container,
                    results_container,
                    error_banner,
                    starter_row,
                    stage_progress_html,
                ],
            )

        return demo

    def launch(self) -> None:
        server_name = os.getenv("MEETING_NOTES_HOST", "127.0.0.1")
        server_port = self.find_available_port(7860, 7870, host=server_name)
        self.demo.launch(
            css=CSS,
            theme=THEME,
            server_name=server_name,
            server_port=server_port,
            share=False,
            show_error=True,
        )
