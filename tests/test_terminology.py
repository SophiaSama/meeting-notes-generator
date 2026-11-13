"""Automated terminology and glossary conformance test suite."""

from __future__ import annotations

from pathlib import Path
import pytest

from meeting_notes.ui import MeetingNotesApp
from meeting_notes.views import (
    STARTER_QUESTIONS,
    classify_error,
    map_progress_to_stage,
    render_empty_state,
)


class TestGradioUITerminology:
    """Validate all visible UI component labels in MeetingNotesApp."""

    @pytest.fixture
    def app(self):
        return MeetingNotesApp()

    def test_app_title_and_header(self, app: MeetingNotesApp):
        assert app.demo.title == "Meeting Notes Helper"

    def test_provider_settings_terminology(self, app: MeetingNotesApp):
        # Accordion label must be 'Provider Settings', not 'LLM Provider Settings'
        assert app.provider_accordion.label == "Provider Settings"
        assert "LLM Provider Settings" not in app.provider_accordion.label

    def test_input_component_labels(self, app: MeetingNotesApp):
        assert app.video_input.label == "Recording"
        assert app.transcript_input.label == "Transcript (optional, skips Recording processing)"

    def test_notes_style_and_language_labels(self, app: MeetingNotesApp):
        blocks_dict = app.demo.blocks
        labels = [getattr(b, "label", None) for b in blocks_dict.values()]
        assert "Notes Style" in labels
        assert "Meeting Notes language" in labels
        assert "Output language" not in labels

    def test_generate_and_reset_button_labels(self, app: MeetingNotesApp):
        assert app.generate_btn.value == "Generate Meeting Notes"
        assert app.reset_btn.value == "Reset"

    def test_results_pane_labels_and_tabs(self, app: MeetingNotesApp):
        assert app.screenshots_gallery.label == "Screenshots"
        assert app.copy_md_btn.value == "Copy Markdown"
        assert getattr(app.download_btn, "label", None) == "Download HTML"
        assert app.html_notes.label == "Rendered Meeting Notes"

    def test_chat_component_labels(self, app: MeetingNotesApp):
        assert app.chatbot.label == "Meeting Chat"
        assert "Meeting Notes" in app.chatbot.placeholder
        assert app.chat_input.placeholder == "Ask a question about the meeting..."


class TestViewsHelpersTerminology:
    """Validate presentation strings emitted by pure view helpers."""

    def test_empty_state_copy(self):
        html_str = render_empty_state()
        assert "No Meeting Notes yet" in html_str
        assert "Add a Recording or Transcript to begin" in html_str
        assert "video" not in html_str.lower()

    def test_canonical_stages(self):
        stages = {
            map_progress_to_stage(0.1, "metadata"),
            map_progress_to_stage(0.4, "transcribing"),
            map_progress_to_stage(0.7, "summarizing"),
            map_progress_to_stage(1.0, "done"),
        }
        assert stages == {"Extracting Screenshots", "Transcribing", "Summarizing", "Done"}

    def test_error_banner_vocabulary(self):
        _cause, fix = classify_error("401 Unauthorized")
        assert "Provider Settings" in fix
        _cause, fix = classify_error("file too large")
        assert "screenshot" in fix.lower()

    def test_starter_questions(self):
        assert "What were the action items?" in STARTER_QUESTIONS
        assert "Who owns what?" in STARTER_QUESTIONS


class TestReadmeTerminology:
    """Validate README.md against obsolete and canonical terms."""

    @pytest.fixture
    def readme_text(self) -> str:
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        return readme_path.read_text(encoding="utf-8")

    def test_no_obsolete_ui_labels_in_readme(self, readme_text: str):
        forbidden_phrases = [
            "LLM Provider Settings",
            "Markdown Notes",
            "HTML Preview",
            "**Video Recording**",
            "Meeting Notes Agent",
            "Captured Screenshots",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in readme_text, f"Obsolete phrase '{phrase}' found in README.md"

    def test_canonical_ui_labels_in_readme(self, readme_text: str):
        required_phrases = [
            "Meeting Notes Helper",
            "Provider Settings",
            "Rendered Meeting Notes",
            "Copy Markdown",
            "Download HTML",
            "Meeting Chat",
            "Extracting Screenshots",
            "Transcribing",
            "Summarizing",
            "Notes Style",
            "Add a Recording or Transcript to begin",
        ]
        for phrase in required_phrases:
            assert phrase in readme_text, f"Required canonical phrase '{phrase}' missing from README.md"
