import os
import socket

from .bootstrap import ProxyDefaults, setup_runtime_environment

setup_runtime_environment()

import gradio as gr

from .llm import REASONING_EFFORT_CHOICES, LLMService
from .notes import NotesService
from .pipeline import PipelineService
from .video import VideoService

CSS = """
.title-row { margin-bottom: .25rem; }
.subtitle   { font-size: 13px; color: #666; margin-top: 2px; }
.note-box   { font-family: monospace; font-size: 12px; }
.capture-hint-box {
    padding: 8px 10px;
    border: 1px solid #8aa8d6;
    background: #eef5ff;
    border-radius: 6px;
    color: #172b4d !important;
    font-size: 13px;
    line-height: 1.45;
}
.capture-hint-box,
.capture-hint-box * {
    color: #172b4d !important;
}
.capture-hint-box strong {
    color: #0f1f3d !important;
    font-weight: 700;
}
footer      { display: none !important; }
"""


class MeetingNotesApp:
    """UI composition and app lifecycle."""

    def __init__(self):
        self.defaults = ProxyDefaults.from_env()
        self.llm_service = LLMService()
        self.video_service = VideoService(self.llm_service)
        self.notes_service = NotesService()
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
            f"for a {self.video_service.format_duration(duration)} recording. "
            f"Estimated notes request: {estimated_mb:.1f} MB of {budget_mb:.1f} MB "
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
        with gr.Blocks(title="Meeting Notes Agent") as demo:
            with gr.Row(elem_classes="title-row"):
                gr.Markdown("# Meeting Notes Agent")
            gr.Markdown(
                "Upload a meeting recording or a transcript file -> generate structured notes via any OpenAI-compatible model.",
                elem_classes="subtitle",
            )

            with gr.Accordion("LLM Provider Settings", open=True):
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
                    self.llm_service.test_connection,
                    inputs=[
                        base_url_input,
                        api_key_input,
                        model_input,
                        reasoning_effort_input,
                    ],
                    outputs=test_result,
                )

                gr.Markdown(self.defaults.status_markdown(), elem_classes="subtitle")

            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.File(
                        label="Video Recording",
                        file_types=["video"],
                        type="filepath",
                    )
                    transcript_input = gr.File(
                        label="Transcript File (optional, skips video processing)",
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
                            label="Notes style",
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
                            label="Output language",
                        )

                    generate_btn = gr.Button("Generate Meeting Notes", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")

                with gr.Column(scale=2):
                    dup_banner = gr.HTML(visible=False)
                    screenshots_gallery = gr.Gallery(
                        label="Captured Screenshots",
                        columns=4,
                        height=160,
                        show_label=True,
                    )
                    with gr.Tabs():
                        with gr.Tab("Markdown Notes"):
                            notes_output = gr.Textbox(
                                label="",
                                lines=20,
                                buttons=["copy"],
                                elem_classes="note-box",
                            )
                        with gr.Tab("HTML Preview"):
                            html_preview = gr.HTML(label="")
                        with gr.Tab("Download"):
                            download_file = gr.File(label="Download HTML notes")

            generate_btn.click(
                self.pipeline_service.run_pipeline_safe,
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
                outputs=[screenshots_gallery, notes_output, html_preview, download_file],
            )

            video_input.change(
                self._on_video_input_change,
                inputs=[video_input],
                outputs=[interval_slider, max_frames_slider, capture_hint, dup_banner],
                show_progress="hidden",
            )

            reset_btn.click(
                lambda: (
                    None,
                    None,
                    [],
                    "",
                    "",
                    None,
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                ),
                inputs=[],
                outputs=[video_input, transcript_input, screenshots_gallery, notes_output, html_preview, download_file, dup_banner, capture_hint],
            )

        return demo

    def launch(self) -> None:
        server_name = os.getenv("MEETING_NOTES_HOST", "127.0.0.1")
        server_port = self.find_available_port(7860, 7870, host=server_name)
        self.demo.launch(
            css=CSS,
            server_name=server_name,
            server_port=server_port,
            share=False,
            show_error=True,
        )
