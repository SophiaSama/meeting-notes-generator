"""Unit and seam tests for MeetingNotesApp handler functions and UI event graph wiring."""

from __future__ import annotations

import re
import subprocess
from unittest.mock import MagicMock

import gradio as gr
import pytest

from meeting_notes.ui import MeetingNotesApp, StageProgressTracker
from meeting_notes.views import (
    format_obsidian_export_success,
    format_obsidian_overwrite_warning,
    format_obsidian_prepare_status,
)
from tests.conftest import FakeLLMService


def _find_block_fn(app: MeetingNotesApp, method_name: str):
    for fn_obj in app.demo.fns.values():
        fn = getattr(fn_obj, "fn", None)
        if fn is not None and getattr(fn, "__name__", "") == method_name:
            return fn_obj
    raise AssertionError(f"Could not find Gradio function for {method_name}")


class TestOnTestConnectionHandler:
    """Test suite for MeetingNotesApp._on_test_connection handler method."""

    def test_on_test_connection_success(self):
        """When test_connection succeeds, returns (res, chip_html) showing 'Connected: <model>'."""
        fake_llm = FakeLLMService(reply="ok")
        app = MeetingNotesApp(llm_service=fake_llm)

        res, chip_html, accordion_update = app._on_test_connection(
            base_url="https://api.openai.com/v1",
            api_key="sk-test-valid-key",
            model="gpt-4o-mini",
            reasoning_effort="none",
        )

        # 1. Connection result text format
        assert isinstance(res, str)
        assert res.startswith("✅")
        assert "ok" in res

        # 2. Updated provider chip HTML format
        assert isinstance(chip_html, str)
        assert "provider-chip" in chip_html
        assert "connected" in chip_html
        assert "unconfigured" not in chip_html
        assert "Connected: gpt-4o-mini" in chip_html
        assert "Not configured" not in chip_html
        assert accordion_update.get("open") is False
        assert "provider-accordion-unconfigured" not in (accordion_update.get("elem_classes") or [])

    def test_on_test_connection_failure(self):
        """When test_connection fails, returns (res, chip_html) showing 'Not configured'."""
        fake_llm = FakeLLMService()
        fake_llm.call_llm = MagicMock(side_effect=RuntimeError("401 Unauthorized: Invalid API key"))
        app = MeetingNotesApp(llm_service=fake_llm)

        res, chip_html, accordion_update = app._on_test_connection(
            base_url="https://api.openai.com/v1",
            api_key="sk-invalid-key",
            model="gpt-4o-mini",
            reasoning_effort="none",
        )

        # 1. Connection result text format
        assert isinstance(res, str)
        assert res.startswith("❌")
        assert "401 Unauthorized" in res

        # 2. Updated provider chip HTML format
        assert isinstance(chip_html, str)
        assert "provider-chip" in chip_html
        assert "unconfigured" in chip_html
        assert "connected" not in chip_html
        assert "Not configured" in chip_html
        assert "Connected:" not in chip_html
        assert accordion_update.get("open") is True
        assert "provider-accordion-unconfigured" in (accordion_update.get("elem_classes") or [])

    def test_on_test_connection_passes_reasoning_effort(self):
        """Handler forwards reasoning_effort to LLMService.test_connection."""
        fake_llm = FakeLLMService(reply="ok")
        app = MeetingNotesApp(llm_service=fake_llm)

        res, chip_html, accordion_update = app._on_test_connection(
            base_url="https://api.openai.com/v1",
            api_key="sk-test-key",
            model="o3-mini",
            reasoning_effort="high",
        )

        assert res.startswith("✅")
        assert "Connected: o3-mini" in chip_html


class TestAccordionAndChipInitialState:
    """Test suite for initial accordion collapse and chip state based on environment."""

    def test_initial_state_when_configured(self, monkeypatch):
        """When API key is present in environment: accordion is collapsed (open=False)
        without highlight class, and chip shows 'Connected: <model>'.
        """
        monkeypatch.setenv("LLM_API_KEY", "sk-live-preconfigured-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

        app = MeetingNotesApp()

        # Accordion is collapsed and unhighlighted
        assert hasattr(app, "provider_accordion")
        assert app.provider_accordion.open is False
        classes = app.provider_accordion.elem_classes or []
        assert "provider-accordion-unconfigured" not in classes

        # Chip reflects configured model
        assert hasattr(app, "provider_status_chip")
        assert "Connected: gpt-4o-mini" in app.provider_status_chip.value
        assert "connected" in app.provider_status_chip.value
        assert "unconfigured" not in app.provider_status_chip.value

    def test_initial_state_when_unconfigured(self, monkeypatch):
        """When API key is absent: accordion is open (open=True) with highlight class,
        and chip shows 'Not configured'.
        """
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "")

        app = MeetingNotesApp()

        # Accordion is open and highlighted
        assert hasattr(app, "provider_accordion")
        assert app.provider_accordion.open is True
        classes = app.provider_accordion.elem_classes or []
        assert any("unconfigured" in c for c in classes)

        # Chip reflects unconfigured state
        assert hasattr(app, "provider_status_chip")
        assert "Not configured" in app.provider_status_chip.value
        assert "unconfigured" in app.provider_status_chip.value
        assert "Connected:" not in app.provider_status_chip.value


class TestEventGraphWiring:
    """Test suite for Gradio event graph wiring and component output counts."""

    def test_test_connection_event_wiring_outputs_count_is_three(self):
        """Verify that test_btn.click is wired to _on_test_connection with exactly 3 outputs:
        [test_result, provider_status_chip, provider_accordion].
        """
        app = MeetingNotesApp()
        test_fn = _find_block_fn(app, "_on_test_connection")

        assert len(test_fn.inputs) == 4
        assert [c._id for c in test_fn.inputs] == [
            app.base_url_input._id,
            app.api_key_input._id,
            app.model_input._id,
            app.reasoning_effort_input._id,
        ]

        assert len(test_fn.outputs) == 3
        assert [c._id for c in test_fn.outputs] == [
            app.test_result._id,
            app.provider_status_chip._id,
            app.provider_accordion._id,
        ]

    def test_existing_lifecycle_event_wiring_remains_unaffected(self):
        """Ensure lifecycle event listeners (_on_clear_chat, _on_generate_start, _on_generate_click, _on_reset)
        have their exact expected component bindings.
        """
        app = MeetingNotesApp()

        clear_fn = _find_block_fn(app, "_on_clear_chat")
        assert len(clear_fn.outputs) == 2

        start_fn = _find_block_fn(app, "_on_generate_start")
        assert len(start_fn.outputs) == 5
        assert [c._id for c in start_fn.outputs] == [
            app.empty_state_container._id,
            app.results_container._id,
            app.error_banner._id,
            app.stage_progress_html._id,
            app.generate_btn._id,
        ]

        gen_fn = _find_block_fn(app, "_on_generate_click")
        assert len(gen_fn.outputs) == 12
        assert app.error_banner._id in [c._id for c in gen_fn.outputs]
        assert app.starter_row._id in [c._id for c in gen_fn.outputs]
        assert app.stage_progress_html._id in [c._id for c in gen_fn.outputs]

        reset_fn = _find_block_fn(app, "_on_reset")
        assert len(reset_fn.outputs) == 16
        assert app.error_banner._id in [c._id for c in reset_fn.outputs]
        assert app.starter_row._id in [c._id for c in reset_fn.outputs]
        assert app.stage_progress_html._id in [c._id for c in reset_fn.outputs]

    def test_generate_finish_and_reset_controls_event_wiring(self):
        """Verify chained .then handlers for _on_generate_finish and _on_reset_controls."""
        app = MeetingNotesApp()

        finish_fn = _find_block_fn(app, "_on_generate_finish")
        assert [c._id for c in finish_fn.inputs] == [app.screenshots_gallery._id]
        assert [c._id for c in finish_fn.outputs] == [
            app.generate_btn._id,
            app.screenshots_tab._id,
            app.results_tabs._id,
        ]

        sync_fn = _find_block_fn(app, "_sync_screenshots_tab")
        assert [c._id for c in sync_fn.inputs] == [app.screenshots_gallery._id]
        assert [c._id for c in sync_fn.outputs] == [app.screenshots_tab._id]
        assert app._sync_screenshots_tab(["s1.png"]).get("visible") is True
        assert app._sync_screenshots_tab([]).get("visible") is False

        reset_ctrl_fn = _find_block_fn(app, "_on_reset_controls")
        assert reset_ctrl_fn.inputs == []
        assert [c._id for c in reset_ctrl_fn.outputs] == [
            app.generate_btn._id,
            app.screenshots_tab._id,
            app.results_tabs._id,
            app.interval_slider._id,
            app.max_frames_slider._id,
        ]

    def test_media_input_change_event_wiring(self):
        """Verify both video_input.change and transcript_input.change are wired."""
        app = MeetingNotesApp()

        video_fn = _find_block_fn(app, "_on_video_input_change")
        assert (app.video_input._id, "change") in video_fn.targets
        assert [c._id for c in video_fn.inputs] == [
            app.video_input._id,
            app.transcript_input._id,
        ]
        assert [c._id for c in video_fn.outputs] == [
            app.interval_slider._id,
            app.max_frames_slider._id,
            app.capture_hint._id,
            app.dup_banner._id,
        ]

        tx_fn = _find_block_fn(app, "_on_transcript_input_change")
        assert (app.transcript_input._id, "change") in tx_fn.targets
        assert [c._id for c in tx_fn.inputs] == [
            app.video_input._id,
            app.transcript_input._id,
        ]
        assert [c._id for c in tx_fn.outputs] == [
            app.interval_slider._id,
            app.max_frames_slider._id,
            app.capture_hint._id,
        ]

    def test_theme_persistence_js_and_compact_share_buttons(self):
        """Verify inline header chip, localStorage theme load/toggle JS, Screenshots tab, and compact share buttons."""
        from meeting_notes.ui import INIT_THEME_JS, THEME_HEAD_HTML, TOGGLE_THEME_JS

        app = MeetingNotesApp()

        assert app.copy_md_btn.scale == 0
        assert app.download_btn.scale == 0
        assert app.screenshots_tab.label == "Screenshots"
        assert app.screenshots_tab.visible is False
        assert app.theme_toggle_btn.value == "☀️ Light"

        # Provider chip sits inline inside .title-row alongside theme_toggle_btn
        assert app.provider_status_chip.parent is app.theme_toggle_btn.parent
        assert "title-row" in (app.provider_status_chip.parent.elem_classes or [])

        # demo.load wires localStorage restoration JS
        load_fns = [
            fn_obj
            for fn_obj in app.demo.fns.values()
            if getattr(fn_obj, "targets", None) == [(app.demo._id, "load")]
        ]
        assert len(load_fns) >= 1
        assert load_fns[0].js is not None
        assert "localStorage" in load_fns[0].js
        assert "localStorage" in THEME_HEAD_HTML
        assert "localStorage" in INIT_THEME_JS
        assert "localStorage" in TOGGLE_THEME_JS
        assert "☀️ Light" in TOGGLE_THEME_JS and "🌙 Dark" in TOGGLE_THEME_JS


class TestResultsPaneEmptyAndPopulatedState:
    """Test suite for Ticket 03: Results pane empty state, results container, and quick share buttons."""

    def test_results_pane_initial_visibility_and_bindings(self):
        """Initial state: empty state is visible, results container is hidden,
        and download/copy buttons are exposed with backward compatibility aliases.
        """
        app = MeetingNotesApp()

        # Component existence and initial visibility
        assert hasattr(app, "empty_state_container")
        assert app.empty_state_container.visible is True
        assert "empty-state-card" in app.empty_state_container.value
        assert "No Meeting Notes yet" in app.empty_state_container.value
        assert "Add a Recording or Transcript to begin" in app.empty_state_container.value

        assert hasattr(app, "results_container")
        assert app.results_container.visible in (False, "hidden")

        # Button components
        assert hasattr(app, "copy_md_btn")
        assert "copy-md-btn" in (app.copy_md_btn.elem_classes or [])

        assert hasattr(app, "download_btn")
        assert "download-btn" in (app.download_btn.elem_classes or [])

        # Compatibility aliases
        assert app.download_file is app.download_btn
        assert app.html_preview is app.html_notes
        assert app.notes_output is app.notes_text

    def test_generate_handler_visibility_transition(self):
        """_on_generate_click returns 12 outputs, hiding empty state, revealing results container, and hiding error banner."""
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = (
            [],
            "# Notes",
            "<p>Notes</p>",
            "test.html",
            {"markdown_notes": "# Notes"},
        )
        mock_pipeline._is_valid_meeting_context.return_value = True
        app.pipeline_service = mock_pipeline

        outputs = app._on_generate_click(
            None,
            "http://test",
            "key",
            "model",
            30,
            8,
            "Summary",
            "English",
        )
        assert len(outputs) == 12
        empty_state_update = outputs[7]
        results_container_update = outputs[8]
        error_banner_update = outputs[9]
        starter_row_update = outputs[10]

        assert empty_state_update.get("visible") is False
        assert results_container_update.get("visible") is True
        assert error_banner_update.get("visible") is False
        assert error_banner_update.get("value") == ""
        assert starter_row_update.get("visible") is True

    def test_reset_handler_visibility_transition(self):
        """_on_reset returns 16 outputs, restoring empty state to visible, hiding results container, error banner, and starter row."""
        app = MeetingNotesApp()
        outputs = app._on_reset()
        assert len(outputs) == 16

        download_val = outputs[5]
        empty_state_update = outputs[11]
        results_container_update = outputs[12]
        error_banner_update = outputs[13]
        starter_row_update = outputs[14]

        assert download_val is None
        assert empty_state_update.get("visible") is True
        assert results_container_update.get("visible") is False
        assert error_banner_update.get("visible") is False
        assert error_banner_update.get("value") == ""
        assert starter_row_update.get("visible") is False

    def test_copy_markdown_button_event_wiring(self):
        """copy_md_btn has a client-side click event wired to clipboard JS."""
        app = MeetingNotesApp()
        copy_fns = [
            fn_obj
            for fn_obj in app.demo.fns.values()
            if getattr(fn_obj, "inputs", None) == [app.notes_text]
        ]
        assert len(copy_fns) >= 1
        copy_fn = copy_fns[0]
        assert copy_fn.fn is None
        assert copy_fn.js is not None
        assert "clipboard" in copy_fn.js


class TestStageProgressTracker:
    """Test suite for StageProgressTracker adapter class."""

    def test_tracker_initialization(self):
        tracker = StageProgressTracker()
        assert tracker.start_time > 0
        assert tracker.current_stage is None
        assert tracker.elapsed_seconds == 0.0
        assert tracker.history == []

    def test_tracker_video_pipeline_callback_sequence(self, monkeypatch):
        mock_progress = MagicMock()
        mock_times = [100.0, 105.0, 125.0, 170.0, 225.0]
        monkeypatch.setattr("time.monotonic", lambda: mock_times.pop(0))

        tracker = StageProgressTracker(progress_fn=mock_progress, is_transcript_only=False)

        # 1. 0.05 metadata
        tracker(0.05, desc="Reading video metadata...")
        assert tracker.current_stage == "Extracting Screenshots"
        mock_progress.assert_called_with(0.05, desc="Extracting Screenshots (5s)")

        # 2. 0.40 transcribing
        tracker(0.40, desc="Transcribing chunk 1...")
        assert tracker.current_stage == "Transcribing"
        mock_progress.assert_called_with(0.40, desc="Transcribing (25s)")

        # 3. 0.62 LLM
        tracker(0.62, desc="Calling gpt-4o...")
        assert tracker.current_stage == "Summarizing"
        mock_progress.assert_called_with(0.62, desc="Summarizing (1m 10s)")

        # 4. 1.0 Done
        tracker(1.0, desc="Done!")
        assert tracker.current_stage == "Done"
        mock_progress.assert_called_with(1.0, desc="Done (2m 05s)")

    def test_tracker_transcript_only_skips_video_stages(self):
        mock_progress = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_progress, is_transcript_only=True)

        tracker(0.10, desc="Reading uploaded transcript...")
        assert tracker.current_stage == "Summarizing"
        assert "Summarizing" in mock_progress.call_args[1]["desc"]

        tracker(0.62, desc="Calling gpt-4o...")
        assert tracker.current_stage == "Summarizing"

        tracker(1.0, desc="Done!")
        assert tracker.current_stage == "Done"

        # Verify Extracting Screenshots and Transcribing were never reached
        stages = [entry["stage"] for entry in tracker.history]
        assert "Extracting Screenshots" not in stages
        assert "Transcribing" not in stages

    def test_tracker_none_callback_graceful(self):
        tracker = StageProgressTracker(progress_fn=None)
        tracker(0.50, desc="Processing...")
        assert tracker.current_stage == "Transcribing"
        assert len(tracker.history) == 1

    def test_tracker_delegates_extra_attributes(self):
        mock_progress = MagicMock()
        mock_progress.tqdm = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_progress)
        assert tracker.tqdm is mock_progress.tqdm

    def test_tracker_handles_positional_only_progress_callable(self):
        calls = []

        def positional_only_fn(fraction):
            calls.append(fraction)

        tracker = StageProgressTracker(progress_fn=positional_only_fn)
        tracker(0.75, desc="Summarizing...")
        assert tracker.current_stage == "Summarizing"
        assert calls == [0.75]


class TestGenerateHandlerStageProgressIntegration:
    """Test suite verifying _on_generate_click instantiates and wires StageProgressTracker."""

    def test_generate_handler_wires_tracker_with_transcript_flag(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = ([], "notes", "html", "out.html", None)
        mock_pipeline._is_valid_meeting_context.return_value = False
        app.pipeline_service = mock_pipeline

        # Run with transcript only
        app._on_generate_click(
            None, "http://localhost", "key", "model", 30, 8, "Summary", "English",
            transcript_file="test.txt",
        )
        _, kwargs = mock_pipeline.run_pipeline_safe.call_args
        passed_tracker = kwargs["progress"]
        assert isinstance(passed_tracker, StageProgressTracker)
        assert passed_tracker.is_transcript_only is True

        # Run with video
        app._on_generate_click(
            "test.mp4", "http://localhost", "key", "model", 30, 8, "Summary", "English",
            transcript_file=None,
        )
        _, kwargs = mock_pipeline.run_pipeline_safe.call_args
        passed_tracker = kwargs["progress"]
        assert isinstance(passed_tracker, StageProgressTracker)
        assert passed_tracker.is_transcript_only is False


class TestErrorBannerLifecycle:
    """Test suite for Ticket 05: Actionable and dismissible error banner UI lifecycle."""

    def test_error_banner_component_properties(self):
        app = MeetingNotesApp()
        assert hasattr(app, "error_banner")
        assert app.error_banner.visible in (False, "hidden")
        assert "error-banner-container" in (app.error_banner.elem_classes or [])

    def test_generate_handler_known_failure_shows_actionable_banner(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = (
            [],
            "Generation failed. You can upload again and retry.\n\nError: RuntimeError: 401 Unauthorized: Invalid API key",
            "",
            None,
            None,
        )
        mock_pipeline._is_valid_meeting_context.return_value = False
        app.pipeline_service = mock_pipeline

        outputs = app._on_generate_click(
            "test.mp4",
            "https://api.openai.com/v1",
            "bad-key",
            "gpt-4o-mini",
            30,
            8,
            "Executive Summary",
            "English",
        )
        assert len(outputs) == 12
        assert outputs[4] is None  # meeting_context
        assert outputs[7]["visible"] is True  # empty_state_container visible
        assert outputs[8]["visible"] is False  # results_container hidden
        banner_update = outputs[9]
        assert banner_update["visible"] is True
        assert "Invalid or missing API key." in banner_update["value"]
        assert "Provider Settings" in banner_update["value"]
        assert "error-banner-dismiss" in banner_update["value"]
        assert outputs[10]["visible"] is False

    def test_generate_handler_exception_shows_actionable_banner(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.side_effect = RuntimeError(
            "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"
        )
        app.pipeline_service = mock_pipeline

        outputs = app._on_generate_click(
            "test.mp4",
            "https://api.openai.com/v1",
            "key",
            "model",
            30,
            8,
            "Summary",
            "English",
        )
        assert len(outputs) == 12
        assert outputs[4] is None
        assert outputs[7]["visible"] is True
        assert outputs[8]["visible"] is False
        banner_update = outputs[9]
        assert banner_update["visible"] is True
        assert "FFmpeg is not installed" in banner_update["value"]
        assert "PATH" in banner_update["value"]
        assert outputs[10]["visible"] is False

    def test_generate_handler_generic_error_shows_troubleshooting_hint(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = (
            [],
            "Generation failed. You can upload again and retry.\n\nError: CustomConnectionLost: peer disconnected abruptly",
            "",
            None,
            None,
        )
        mock_pipeline._is_valid_meeting_context.return_value = False
        app.pipeline_service = mock_pipeline

        outputs = app._on_generate_click(
            "test.mp4",
            "https://api.openai.com/v1",
            "key",
            "model",
            30,
            8,
            "Summary",
            "English",
        )
        assert len(outputs) == 12
        banner_update = outputs[9]
        assert banner_update["visible"] is True
        assert "CustomConnectionLost" in banner_update["value"]
        assert "application logs" in banner_update["value"]
        assert outputs[10]["visible"] is False

    def test_reset_handler_clears_error_banner(self):
        app = MeetingNotesApp()
        outputs = app._on_reset()
        assert len(outputs) == 16
        error_banner_update = outputs[13]
        assert error_banner_update.get("visible") is False
        assert error_banner_update.get("value") == ""
        assert outputs[14].get("visible") is False


class TestChatStarterChipsLifecycle:
    """Test suite for Ticket 06: Meeting Chat starter question chips and visibility lifecycle."""

    def test_starter_row_initial_state_hidden(self):
        """Initial state: starter_row exists, is hidden (visible=False), and has chat-starters-row class."""
        from meeting_notes.views import STARTER_QUESTIONS

        app = MeetingNotesApp()
        assert hasattr(app, "starter_row")
        assert app.starter_row.visible is False
        classes = app.starter_row.elem_classes or []
        assert "chat-starters-row" in classes or "starter-chips-row" in classes

        assert hasattr(app, "starter_buttons")
        assert len(app.starter_buttons) == len(STARTER_QUESTIONS)
        rendered_labels = [b.value for b in app.starter_buttons]
        assert rendered_labels == STARTER_QUESTIONS
        assert "What were the action items?" in rendered_labels
        assert "Who owns what?" in rendered_labels

    def test_starter_prefill_handler_returns_update(self):
        """MeetingNotesApp._on_starter_click returns a Gradio update setting value without auto-sending."""
        app = MeetingNotesApp()
        update = app._on_starter_click("What were the action items?")
        assert update.get("value") == "What were the action items?"

    def test_starter_buttons_event_wiring_outputs_only_to_chat_input(self):
        """Each starter chip button is wired to output ONLY to chat_input (inputs=[], outputs=[chat_input])."""
        from meeting_notes.views import STARTER_QUESTIONS

        app = MeetingNotesApp()

        starter_fns = [
            fn_obj
            for fn_obj in app.demo.fns.values()
            if getattr(fn_obj, "inputs", None) == []
            and [c._id for c in getattr(fn_obj, "outputs", [])] == [app.chat_input._id]
        ]
        assert len(starter_fns) == len(STARTER_QUESTIONS)

        for fn_obj in starter_fns:
            assert app.chatbot._id not in [c._id for c in fn_obj.outputs]
            assert app.meeting_context_state._id not in [c._id for c in fn_obj.outputs]

    def test_generate_success_reveals_starter_row(self):
        """On successful notes generation, _on_generate_click sets starter_row visible=True."""
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = (
            ["thumb.png"],
            "# Meeting Notes",
            "<p>Notes</p>",
            "out.html",
            {"markdown_notes": "# Meeting Notes"},
        )
        mock_pipeline._is_valid_meeting_context.return_value = True
        app.pipeline_service = mock_pipeline

        outputs = app._on_generate_click(
            "test.mp4", "https://api.openai.com", "key", "model", 30, 8, "Summary", "English"
        )
        assert len(outputs) == 12
        starter_row_update = outputs[10]
        assert starter_row_update.get("visible") is True

    def test_generate_failure_hides_starter_row(self):
        """On failed notes generation, _on_generate_click sets starter_row visible=False."""
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = ([], "Error occurred", "", None, None)
        mock_pipeline._is_valid_meeting_context.return_value = False
        app.pipeline_service = mock_pipeline

        outputs = app._on_generate_click(
            "test.mp4", "https://api.openai.com", "bad-key", "model", 30, 8, "Summary", "English"
        )
        assert len(outputs) == 12
        starter_row_update = outputs[10]
        assert starter_row_update.get("visible") is False

    def test_reset_hides_starter_row(self):
        """On reset, _on_reset sets starter_row visible=False."""
        app = MeetingNotesApp()
        outputs = app._on_reset()
        assert len(outputs) == 16
        starter_row_update = outputs[14]
        assert starter_row_update.get("visible") is False


    def test_generate_start_switches_to_generating_state(self):
        """_on_generate_start hides empty/results/error, shows 4-step stage card, and locks Generate button."""
        app = MeetingNotesApp()
        empty_u, results_u, err_u, stage_u, btn_u = app._on_generate_start(video_file="meeting.mp4")
        assert empty_u.get("visible") is False
        assert results_u.get("visible") is False
        assert err_u.get("visible") is False
        assert stage_u.get("visible") is True
        assert "Extracting Screenshots" in stage_u.get("value", "")
        assert "stage-progress-card" in stage_u.get("value", "")
        assert btn_u.get("value") == "Generating..."
        assert btn_u.get("interactive") is False

        _, _, _, stage_tx, btn_tx = app._on_generate_start(video_file=None, transcript_file="notes.txt")
        assert stage_tx.get("visible") is True
        assert "Summarizing" in stage_tx.get("value", "")
        assert 'data-stage="Extracting Screenshots" data-state="skipped"' in stage_tx.get("value", "")
        assert 'data-stage="Transcribing" data-state="skipped"' in stage_tx.get("value", "")
        assert btn_tx.get("value") == "Generating..."
        assert btn_tx.get("interactive") is False


class TestGenerateButtonLockAndScreenshotsTab:
    """Test suite for button locking/unlocking and conditional Screenshots tab visibility."""

    def test_initial_controls_state(self):
        app = MeetingNotesApp()
        assert app.generate_btn.value == "Generate Meeting Notes"
        assert app.generate_btn.interactive is True
        assert app.screenshots_tab.visible is False
        assert app.interval_slider.visible is True
        assert app.max_frames_slider.visible is True

    def test_generate_finish_with_screenshots_unlocks_button_and_shows_tab(self):
        app = MeetingNotesApp()
        btn_u, tab_u, tabs_u = app._on_generate_finish(thumbnails=["frame_1.png", "frame_2.png"])
        assert btn_u.get("value") == "Generate Meeting Notes"
        assert btn_u.get("interactive") is True
        assert tab_u.get("visible") is True
        assert tabs_u.get("selected") is None

    def test_generate_finish_without_screenshots_unlocks_button_and_hides_tab(self):
        app = MeetingNotesApp()
        for empty_thumbs in ([], None):
            btn_u, tab_u, tabs_u = app._on_generate_finish(thumbnails=empty_thumbs)
            assert btn_u.get("value") == "Generate Meeting Notes"
            assert btn_u.get("interactive") is True
            assert tab_u.get("visible") is False
            assert tabs_u.get("selected") == "rendered-notes"

    def test_stage_progress_tracker_passes_transcript_only_flag_to_card_html(self):
        tracker = StageProgressTracker(progress_fn=None, is_transcript_only=True)
        tracker(0.2, desc="Reading transcript...")
        assert 'data-stage="Extracting Screenshots" data-state="skipped"' in tracker.last_html
        assert 'data-stage="Transcribing" data-state="skipped"' in tracker.last_html
        assert 'data-stage="Summarizing" data-state="active"' in tracker.last_html

    def test_full_generate_chain_recording_vs_transcript_vs_error(self):
        app = MeetingNotesApp(llm_service=FakeLLMService())
        mock_pipeline = MagicMock()
        app.pipeline_service = mock_pipeline

        # 1. Recording run with screenshots
        mock_pipeline.run_pipeline_safe.return_value = (
            ["s1.png"],
            "# Notes",
            "<p>Notes</p>",
            "out.html",
            {"markdown_notes": "# Notes"},
        )
        mock_pipeline._is_valid_meeting_context.return_value = True

        _, _, _, _, btn_start = app._on_generate_start(video_file="rec.mp4", transcript_file=None)
        assert btn_start.get("value") == "Generating..."
        assert btn_start.get("interactive") is False

        gen_out = app._on_generate_click(
            "rec.mp4", "https://api.openai.com/v1", "key", "model", 30, 8, "Executive Summary", "English"
        )
        assert len(gen_out) == 12
        assert gen_out[0] == ["s1.png"]
        assert gen_out[8].get("visible") is True

        btn_fin, tab_fin, _tabs_fin = app._on_generate_finish(gen_out[0])
        assert btn_fin.get("value") == "Generate Meeting Notes"
        assert btn_fin.get("interactive") is True
        assert tab_fin.get("visible") is True

        # 2. Transcript-only run (no screenshots)
        mock_pipeline.run_pipeline_safe.return_value = (
            [],
            "# Notes",
            "<p>Notes</p>",
            "out.html",
            {"markdown_notes": "# Notes"},
        )
        _, _, _, stage_tx, btn_tx = app._on_generate_start(video_file=None, transcript_file="tx.txt")
        assert btn_tx.get("interactive") is False
        assert "Summarizing" in stage_tx.get("value", "")

        tx_out = app._on_generate_click(
            None, "https://api.openai.com/v1", "key", "model", 30, 8, "Executive Summary", "English",
            transcript_file="tx.txt",
        )
        assert len(tx_out) == 12
        assert tx_out[0] == []
        assert tx_out[8].get("visible") is True

        btn_tx_fin, tab_tx_fin, tabs_tx_fin = app._on_generate_finish(tx_out[0])
        assert btn_tx_fin.get("value") == "Generate Meeting Notes"
        assert btn_tx_fin.get("interactive") is True
        assert tab_tx_fin.get("visible") is False
        assert tabs_tx_fin.get("selected") == "rendered-notes"

        # 3. Error run still unlocks Generate button and hides Screenshots tab
        mock_pipeline.run_pipeline_safe.side_effect = RuntimeError("401 Unauthorized")
        err_out = app._on_generate_click(
            "rec.mp4", "https://api.openai.com/v1", "key", "model", 30, 8, "Executive Summary", "English"
        )
        assert len(err_out) == 12
        assert err_out[0] == []
        assert err_out[9].get("visible") is True

        btn_err_fin, tab_err_fin, _ = app._on_generate_finish(err_out[0])
        assert btn_err_fin.get("value") == "Generate Meeting Notes"
        assert btn_err_fin.get("interactive") is True
        assert tab_err_fin.get("visible") is False


class TestCaptureSliderContextVisibility:
    """Test suite for context-sensitive visibility of capture sliders and hint."""

    def test_transcript_only_hides_capture_sliders_and_hint(self):
        app = MeetingNotesApp()
        mock_video = MagicMock()
        app.video_service = mock_video

        interval_u, max_u, hint_u, dup_u = app._on_video_input_change(None, "notes.txt")
        assert interval_u.get("visible") is False
        assert max_u.get("visible") is False
        assert hint_u.get("visible") is False
        assert hint_u.get("value") == ""
        assert dup_u.get("visible") is False

        t_interval_u, t_max_u, t_hint_u = app._on_transcript_input_change(None, "notes.txt")
        assert t_interval_u.get("visible") is False
        assert t_max_u.get("visible") is False
        assert t_hint_u.get("visible") is False
        assert t_hint_u.get("value") == ""
        mock_video.get_video_info.assert_not_called()

    def test_recording_and_transcript_restores_capture_sliders(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.check_upload_for_duplicate.return_value = {"__type__": "update", "value": "", "visible": False}
        mock_pipeline.recommend_capture_settings.return_value = {
            "interval_seconds": 20,
            "max_frames": 6,
            "estimated_request_bytes": 1_048_576,
            "budget_bytes": 10_485_760,
        }
        mock_video = MagicMock()
        mock_video.get_video_info.return_value = {"duration": 120.0, "resolution": "1280x720"}
        mock_video.format_duration.return_value = "2m 00s"
        app.pipeline_service = mock_pipeline
        app.video_service = mock_video
        app._sample_average_frame_request_bytes = MagicMock(return_value=None)

        interval_u, max_u, hint_u, _dup_u = app._on_video_input_change("meeting.mp4", "notes.txt")
        assert interval_u.get("visible") is True
        assert interval_u.get("value") == 20
        assert max_u.get("visible") is True
        assert max_u.get("value") == 6
        assert hint_u.get("visible") is True

        # Adding/removing Transcript while Recording is present preserves capture hint without re-running ffmpeg
        mock_video.get_video_info.reset_mock()
        t_interval_u, t_max_u, t_hint_u = app._on_transcript_input_change("meeting.mp4", "notes.txt")
        assert t_interval_u.get("visible") is True
        assert t_max_u.get("visible") is True
        assert "visible" not in t_hint_u
        mock_video.get_video_info.assert_not_called()

    def test_clearing_transcript_restores_capture_sliders(self):
        app = MeetingNotesApp()
        interval_u, max_u, hint_u, dup_u = app._on_video_input_change(None, None)
        assert interval_u.get("visible") is True
        assert max_u.get("visible") is True
        assert hint_u.get("visible") is False
        assert dup_u.get("visible") is False

        t_interval_u, t_max_u, t_hint_u = app._on_transcript_input_change(None, None)
        assert t_interval_u.get("visible") is True
        assert t_max_u.get("visible") is True
        assert t_hint_u.get("visible") is False

    def test_recording_duplicate_keeps_capture_sliders_visible(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        mock_pipeline.check_upload_for_duplicate.return_value = {
            "__type__": "update",
            "value": "<div>Duplicate</div>",
            "visible": True,
        }
        app.pipeline_service = mock_pipeline

        interval_u, max_u, hint_u, dup_u = app._on_video_input_change("dup.mp4", None)
        assert interval_u.get("visible") is True
        assert max_u.get("visible") is True
        assert hint_u.get("visible") is False
        assert dup_u.get("visible") is True


class TestResetControlsRestoration:
    """Test suite for Reset restoring Generate button, Screenshots tab, active tab, and sliders."""

    def test_on_reset_controls_restores_button_tabs_and_sliders(self):
        app = MeetingNotesApp()
        btn_u, tab_u, tabs_u, interval_u, max_u = app._on_reset_controls()
        assert btn_u.get("value") == "Generate Meeting Notes"
        assert btn_u.get("interactive") is True
        assert tab_u.get("visible") is False
        assert tabs_u.get("selected") == "rendered-notes"
        assert interval_u.get("value") == 30
        assert interval_u.get("visible") is True
        assert max_u.get("value") == 8
        assert max_u.get("visible") is True


class TestObsidianExportViews:
    """Unit tests for pure Obsidian export formatting helpers in meeting_notes/views.py."""

    _FORBIDDEN_GLOSSARY_RE = re.compile(
        r"\b(minutes|frames?|images?|chatbot|llm settings|backend)\b",
        re.IGNORECASE,
    )

    def test_prepare_status_with_screenshots_and_faq_avoids_overwrite_and_glossary_terms(self):
        msg = format_obsidian_prepare_status(
            {
                "target_dir": "/vault/2026-11-24 Weekly Sync",
                "notes_filename": "2026-11-24 Weekly Sync.md",
                "screenshot_count": 3,
                "default_attachment_prefix": "sync",
                "faq_filename": "2026-11-24 Weekly Sync - FAQ.md",
                "faq_markdown": "## FAQ\n\n### Owner?\nAlice.",
            }
        )
        assert "/vault/2026-11-24 Weekly Sync" in msg
        assert "2026-11-24 Weekly Sync.md" in msg
        assert "3 Screenshot(s)" in msg
        assert "2026-11-24 Weekly Sync - FAQ.md" in msg
        assert "already exists" not in msg.lower()
        assert "overwrite" not in msg.lower()
        assert not self._FORBIDDEN_GLOSSARY_RE.search(msg)

    def test_prepare_status_zero_screenshots_and_empty_faq(self):
        msg = format_obsidian_prepare_status(
            {
                "target_dir": "/vault/2026-11-24 Transcript Sync",
                "notes_filename": "2026-11-24 Transcript Sync.md",
                "screenshot_count": 0,
                "default_attachment_prefix": "screenshot",
                "faq_filename": "2026-11-24 Transcript Sync - FAQ.md",
                "faq_markdown": None,
            }
        )
        assert "0 Screenshots" in msg
        assert "no `attachment/` folder" in msg
        assert "omitted" in msg
        assert "already exists" not in msg.lower()
        assert "overwrite" not in msg.lower()
        assert not self._FORBIDDEN_GLOSSARY_RE.search(msg)

    def test_overwrite_warning_and_export_success_copy(self):
        warn = format_obsidian_overwrite_warning("/vault/2026-11-24 Weekly Sync")
        assert "already exists" in warn.lower()
        assert "overwrite" in warn.lower()
        assert "/vault/2026-11-24 Weekly Sync" in warn
        assert not self._FORBIDDEN_GLOSSARY_RE.search(warn)

        success = format_obsidian_export_success(
            {
                "target_dir": "/vault/2026-11-24 Weekly Sync",
                "notes_filename": "2026-11-24 Weekly Sync.md",
                "screenshots_copied": 2,
                "faq_path": "/vault/2026-11-24 Weekly Sync/2026-11-24 Weekly Sync - FAQ.md",
                "faq_filename": "2026-11-24 Weekly Sync - FAQ.md",
            }
        )
        assert "/vault/2026-11-24 Weekly Sync" in success
        assert "2 Screenshot(s)" in success
        assert "2026-11-24 Weekly Sync - FAQ.md" in success
        assert "already exists" not in success.lower()
        assert "overwrite" not in success.lower()
        assert not self._FORBIDDEN_GLOSSARY_RE.search(success)


class TestObsidianExportUIHandlers:
    """Unit tests for Export to Obsidian UI controls, subprocess picker, state transitions, and reset."""

    _FORBIDDEN_GLOSSARY_RE = re.compile(
        r"\b(minutes|frames?|images?|chatbot|llm settings|backend)\b",
        re.IGNORECASE,
    )

    def test_initial_obsidian_controls_state_and_glossary(self):
        app = MeetingNotesApp()
        assert isinstance(app.obsidian_export_state, gr.State)
        assert app.obsidian_export_state.value is None
        assert not hasattr(app, "obsidian_export")
        assert app.obsidian_export_accordion.label == "Export to Obsidian"
        assert app.obsidian_export_accordion.open is False
        assert app.obsidian_vault_input.interactive is True
        assert app.obsidian_prefix_input.value == "screenshot"
        assert app.obsidian_prepare_btn.interactive is False
        assert app.obsidian_confirm_btn.visible in (False, "hidden")
        assert app.obsidian_cancel_btn.visible in (False, "hidden")
        assert app.obsidian_faq_preview.visible in (False, "hidden")
        assert app.obsidian_status_md.visible in (False, "hidden")
        assert app.obsidian_warning_md.visible in (False, "hidden")

        for comp in (
            app.obsidian_export_accordion,
            app.obsidian_vault_input,
            app.obsidian_browse_btn,
            app.obsidian_prefix_input,
            app.obsidian_prepare_btn,
            app.obsidian_status_md,
            app.obsidian_warning_md,
            app.obsidian_faq_preview,
            app.obsidian_confirm_btn,
            app.obsidian_cancel_btn,
        ):
            blob = " ".join(
                str(getattr(comp, attr, "") or "")
                for attr in ("label", "value", "info", "placeholder")
            )
            assert not self._FORBIDDEN_GLOSSARY_RE.search(blob), blob
            assert "already exists" not in blob.lower()
            assert "overwrite" not in blob.lower()

    def test_after_generate_and_reset_sync_guard_and_clear_pending_state(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        app.pipeline_service = mock_pipeline

        mock_pipeline._is_valid_meeting_context.return_value = True
        state_out, prep_u, conf_u, canc_u, faq_u, warn_u, status_u = app._on_obsidian_after_generate(
            {"raw_notes": "# Title", "markdown_notes": "# Title"}
        )
        assert state_out is None
        assert prep_u.get("interactive") is True
        assert conf_u.get("visible") is False
        assert canc_u.get("visible") is False
        assert faq_u.get("value") == "" and faq_u.get("visible") is False
        assert warn_u.get("value") == "" and warn_u.get("visible") is False
        assert status_u.get("value") == "" and status_u.get("visible") is False

        # Empty/invalid context keeps prepare_btn disabled
        mock_pipeline._is_valid_meeting_context.return_value = False
        state_bad, prep_bad, *_ = app._on_obsidian_after_generate(None)
        assert state_bad is None
        assert prep_bad.get("interactive") is False

        # Reset restores defaults and collapses accordion
        r_state, r_vault, r_prefix, r_prep, r_conf, r_canc, r_faq, r_warn, r_status, r_acc = (
            app._on_obsidian_reset()
        )
        assert r_state is None
        assert r_vault.get("value") == ""
        assert r_prefix.get("value") == "screenshot"
        assert r_prep.get("interactive") is False
        assert r_conf.get("visible") is False
        assert r_canc.get("visible") is False
        assert r_faq.get("value") == "" and r_faq.get("visible") is False
        assert r_warn.get("value") == "" and r_warn.get("visible") is False
        assert r_status.get("value") == "" and r_status.get("visible") is False
        assert r_acc.get("open") is False

    def test_browse_subprocess_success_cancel_and_error_fallbacks(self, monkeypatch, tmp_path):
        app = MeetingNotesApp()
        calls: list[tuple[list[str], dict]] = []

        def fake_success(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout=f"  {tmp_path}/ \n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_success)
        vault_u, status_u = app._on_obsidian_browse(str(tmp_path))
        assert len(calls) == 1
        assert calls[0][1].get("shell", False) is False
        assert vault_u.get("value") == str(tmp_path)
        assert status_u.get("visible") is True

        # Cancel keeps existing path via gr.update() (no value key)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="User canceled"),
        )
        vault_cancel, status_cancel = app._on_obsidian_browse("/existing/vault")
        assert "value" not in vault_cancel
        assert status_cancel.get("visible") is True

        # Empty stdout keeps existing path
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="   \n", stderr=""),
        )
        vault_empty, status_empty = app._on_obsidian_browse("/existing/vault")
        assert "value" not in vault_empty
        assert status_empty.get("visible") is True

        # Timeout and missing binary fall back gracefully without raising
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=subprocess.TimeoutExpired(["osascript"], 60)),
        )
        vault_to, status_to = app._on_obsidian_browse("/existing/vault")
        assert "value" not in vault_to
        assert "manually" in status_to.get("value", "").lower()

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=FileNotFoundError("no picker binary")),
        )
        vault_err, status_err = app._on_obsidian_browse("/existing/vault")
        assert "value" not in vault_err
        assert "manually" in status_err.get("value", "").lower()

    def test_build_folder_picker_command_platforms(self, monkeypatch, tmp_path):
        # macOS passes initial_dir via argv
        monkeypatch.setattr("meeting_notes.ui.sys.platform", "darwin")
        cmd_mac, env_mac = MeetingNotesApp._build_folder_picker_command(str(tmp_path))
        assert cmd_mac[0] == "osascript"
        assert cmd_mac[-1] == str(tmp_path)
        assert env_mac is None

        # Windows passes initial_dir via env var
        monkeypatch.setattr("meeting_notes.ui.sys.platform", "win32")
        monkeypatch.setattr("meeting_notes.ui.shutil.which", lambda name: "powershell.exe" if name == "powershell" else None)
        cmd_win, env_win = MeetingNotesApp._build_folder_picker_command(str(tmp_path))
        assert "-STA" in cmd_win
        assert env_win is not None and env_win["MEETING_NOTES_PICKER_INIT_DIR"] == str(tmp_path)

        # Linux falls back to zenity -> kdialog -> tkinter
        monkeypatch.setattr("meeting_notes.ui.sys.platform", "linux")
        monkeypatch.setattr("meeting_notes.ui.shutil.which", lambda name: "/usr/bin/zenity" if name == "zenity" else None)
        cmd_zen, _ = MeetingNotesApp._build_folder_picker_command(str(tmp_path))
        assert cmd_zen[0] == "/usr/bin/zenity"

        monkeypatch.setattr("meeting_notes.ui.shutil.which", lambda name: None)
        cmd_tk, _ = MeetingNotesApp._build_folder_picker_command(str(tmp_path))
        assert cmd_tk[1] == "-c"
        assert cmd_tk[-1] == str(tmp_path)

    def test_browse_and_command_handle_embedded_nul_byte_gracefully(self, monkeypatch):
        bad_path = "~\x00bad"
        cmd, env = MeetingNotesApp._build_folder_picker_command(bad_path)
        assert isinstance(cmd, list) and len(cmd) >= 1
        assert bad_path not in cmd
        if env is not None:
            assert "MEETING_NOTES_PICKER_INIT_DIR" not in env

        app = MeetingNotesApp()
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda c, **kw: subprocess.CompletedProcess(c, 1, stdout="", stderr="canceled"),
        )
        vault_u, status_u = app._on_obsidian_browse(bad_path)
        assert "value" not in vault_u
        assert status_u.get("visible") is True

    @pytest.mark.parametrize("bad_tilde", ["~nonexistent_user_xyz", "~bob/vault"])
    def test_browse_and_command_handle_unknown_tilde_user_gracefully(self, monkeypatch, bad_tilde):
        for platform in ("darwin", "win32", "linux"):
            monkeypatch.setattr("meeting_notes.ui.sys.platform", platform)
            if platform == "win32":
                monkeypatch.setattr(
                    "meeting_notes.ui.shutil.which",
                    lambda name: "powershell.exe" if name == "powershell" else None,
                )
            elif platform == "linux":
                monkeypatch.setattr("meeting_notes.ui.shutil.which", lambda name: None)

            cmd, env = MeetingNotesApp._build_folder_picker_command(bad_tilde)
            assert isinstance(cmd, list) and len(cmd) >= 1
            assert bad_tilde not in cmd
            if env is not None:
                assert "MEETING_NOTES_PICKER_INIT_DIR" not in env

        app = MeetingNotesApp()
        seen_cmds: list[list[str]] = []

        def fake_run(c, **kw):
            seen_cmds.append(c)
            return subprocess.CompletedProcess(c, 1, stdout="", stderr="canceled")

        monkeypatch.setattr(subprocess, "run", fake_run)
        vault_u, status_u = app._on_obsidian_browse(bad_tilde)
        assert len(seen_cmds) == 1
        assert bad_tilde not in seen_cmds[0]
        assert "value" not in vault_u
        assert status_u.get("visible") is True

    def test_build_folder_picker_command_linux_kdialog_and_home_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr("meeting_notes.ui.sys.platform", "linux")
        monkeypatch.setattr(
            "meeting_notes.ui.shutil.which",
            lambda name: "/usr/bin/kdialog" if name == "kdialog" else None,
        )

        # 1. Valid vault path uses initial_dir
        cmd_dir, env_dir = MeetingNotesApp._build_folder_picker_command(str(tmp_path))
        assert cmd_dir == [
            "/usr/bin/kdialog",
            "--getexistingdirectory",
            str(tmp_path),
            "--title",
            "Select Obsidian Vault folder",
        ]
        assert env_dir is None

        # 2. Empty path falls back to Path.home() when available
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr("meeting_notes.ui.Path.home", classmethod(lambda cls: fake_home))
        cmd_home, env_home = MeetingNotesApp._build_folder_picker_command("")
        assert cmd_home == [
            "/usr/bin/kdialog",
            "--getexistingdirectory",
            str(fake_home),
            "--title",
            "Select Obsidian Vault folder",
        ]
        assert env_home is None

        # 3. Unset HOME + Path.home() RuntimeError omits start_dir cleanly
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setattr(
            "meeting_notes.ui.Path.home",
            classmethod(
                lambda cls: (_ for _ in ()).throw(
                    RuntimeError("Could not determine home directory.")
                )
            ),
        )
        cmd_no_home, env_no_home = MeetingNotesApp._build_folder_picker_command("")
        assert cmd_no_home == [
            "/usr/bin/kdialog",
            "--getexistingdirectory",
            "--title",
            "Select Obsidian Vault folder",
        ]
        assert env_no_home is None

        app = MeetingNotesApp()
        seen_cmds: list[list[str]] = []

        def fake_run(c, **kw):
            seen_cmds.append(c)
            return subprocess.CompletedProcess(c, 1, stdout="", stderr="canceled")

        monkeypatch.setattr(subprocess, "run", fake_run)
        vault_u, status_u = app._on_obsidian_browse("")
        assert seen_cmds == [
            [
                "/usr/bin/kdialog",
                "--getexistingdirectory",
                "--title",
                "Select Obsidian Vault folder",
            ]
        ]
        assert "value" not in vault_u
        assert status_u.get("visible") is True

    def test_browse_error_guards_fall_back_to_manual_entry(self, monkeypatch):
        app = MeetingNotesApp()
        with monkeypatch.context() as m:
            run_mock = MagicMock()
            m.setattr(subprocess, "run", run_mock)
            m.setattr(
                app,
                "_build_folder_picker_command",
                MagicMock(side_effect=RuntimeError("picker command failure")),
            )
            vault_u, status_u = app._on_obsidian_browse("/existing/vault")
            assert "value" not in vault_u
            assert status_u.get("visible") is True
            assert "manually" in status_u.get("value", "").lower()
            run_mock.assert_not_called()

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=PermissionError("operation not permitted")),
        )
        vault_perm, status_perm = app._on_obsidian_browse("/existing/vault")
        assert "value" not in vault_perm
        assert status_perm.get("visible") is True
        assert "manually" in status_perm.get("value", "").lower()

    def test_prepare_confirm_and_cancel_state_transitions(self):
        app = MeetingNotesApp()
        mock_pipeline = MagicMock()
        app.pipeline_service = mock_pipeline

        # 1. Invalid meeting_context blocks before calling prepare_obsidian_export
        mock_pipeline._is_valid_meeting_context.return_value = False
        st, status_u, warn_u, faq_u, conf_u, canc_u = app._on_obsidian_prepare(
            "/vault", "shot", None, [], "http://x", "k", "m",
        )
        assert st is None
        assert status_u.get("visible") is True
        assert conf_u.get("visible") is False
        assert canc_u.get("visible") is False
        assert faq_u.get("visible") is False
        mock_pipeline.prepare_obsidian_export.assert_not_called()

        # 2. Prepare exception keeps confirm/cancel/faq hidden
        mock_pipeline._is_valid_meeting_context.return_value = True
        mock_pipeline.prepare_obsidian_export.side_effect = ValueError("Vault path does not exist")
        ctx = {"raw_notes": "# Sync", "markdown_notes": "# Sync"}
        st, status_u, warn_u, faq_u, conf_u, canc_u = app._on_obsidian_prepare(
            "/missing", "shot", ctx, [], "http://x", "k", "m",
        )
        assert st is None
        assert "Vault path does not exist" in status_u.get("value", "")
        assert warn_u.get("visible") is False
        assert faq_u.get("visible") is False
        assert conf_u.get("visible") is False
        assert canc_u.get("visible") is False

        # 3. Prepare with collision & non-empty FAQ reveals warning, editable FAQ, confirm & cancel
        prepared = {
            "vault_path": "/vault",
            "folder_name": "2026-11-24 Sync",
            "target_dir": "/vault/2026-11-24 Sync",
            "notes_filename": "2026-11-24 Sync.md",
            "faq_filename": "2026-11-24 Sync - FAQ.md",
            "exists": True,
            "faq_markdown": "## FAQ\n\n### Q?\nA.",
            "screenshot_count": 1,
            "default_attachment_prefix": "shot",
        }
        mock_pipeline.prepare_obsidian_export.side_effect = None
        mock_pipeline.prepare_obsidian_export.return_value = prepared
        st, status_u, warn_u, faq_u, conf_u, canc_u = app._on_obsidian_prepare(
            "/vault", "shot", ctx, [{"role": "user", "content": "Q?"}], "http://x", "k", "m",
        )
        assert st == prepared
        assert "/vault/2026-11-24 Sync" in status_u.get("value", "")
        assert "already exists" not in status_u.get("value", "").lower()
        assert warn_u.get("visible") is True
        assert "already exists" in warn_u.get("value", "").lower()
        assert faq_u.get("visible") is True
        assert faq_u.get("interactive") is True
        assert faq_u.get("value") == "## FAQ\n\n### Q?\nA."
        assert conf_u.get("visible") is True
        assert canc_u.get("visible") is True

        # 4. Cancel clears state without calling export_to_obsidian
        c_st, c_status, c_warn, c_faq, c_conf, c_canc = app._on_obsidian_cancel()
        assert c_st is None
        assert "canceled" in c_status.get("value", "").lower()
        assert c_warn.get("visible") is False
        assert c_faq.get("visible") is False
        assert c_conf.get("visible") is False
        assert c_canc.get("visible") is False
        mock_pipeline.export_to_obsidian.assert_not_called()

        # 5. Confirm without prepared state does not call export_to_obsidian
        g_st, g_status, _g_warn, _g_faq, g_conf, g_canc = app._on_obsidian_confirm(None, "shot", "")
        assert g_st is None
        assert g_status.get("visible") is True
        assert g_conf.get("visible") is False
        assert g_canc.get("visible") is False
        mock_pipeline.export_to_obsidian.assert_not_called()

        # 6. Confirm conflict updates state exists=True and surfaces warning
        mock_pipeline.export_to_obsidian.return_value = {
            "status": "conflict",
            "written": False,
            "target_dir": "/vault/2026-11-24 Sync",
        }
        non_existing = {**prepared, "exists": False}
        cf_st, cf_status, cf_warn, _cf_faq, cf_conf, cf_canc = app._on_obsidian_confirm(
            non_existing, "shot", "## Edited FAQ",
        )
        assert cf_st["exists"] is True
        assert cf_warn.get("visible") is True
        assert "already exists" in cf_warn.get("value", "").lower()
        assert cf_conf.get("visible") is True
        assert cf_canc.get("visible") is True

        # 7. Confirm success writes, clears state, and hides confirmation widgets
        mock_pipeline.export_to_obsidian.return_value = {
            "status": "exported",
            "written": True,
            "target_dir": "/vault/2026-11-24 Sync",
            "notes_filename": "2026-11-24 Sync.md",
            "screenshots_copied": 1,
            "faq_path": "/vault/2026-11-24 Sync/2026-11-24 Sync - FAQ.md",
            "faq_filename": "2026-11-24 Sync - FAQ.md",
        }
        ok_st, ok_status, ok_warn, ok_faq, ok_conf, ok_canc = app._on_obsidian_confirm(
            prepared, "shot", "## Edited FAQ",
        )
        mock_pipeline.export_to_obsidian.assert_called_with(
            prepared,
            attachment_prefix="shot",
            faq_markdown="## Edited FAQ",
            overwrite=True,
        )
        assert ok_st is None
        assert "Exported Meeting Notes" in ok_status.get("value", "")
        assert ok_warn.get("visible") is False
        assert ok_faq.get("visible") is False
        assert ok_conf.get("visible") is False
        assert ok_canc.get("visible") is False

    def test_wrap_html_preview_preserves_doctype_and_wraps_body_in_article(self):
        full_doc = "<!DOCTYPE html><html><head></head><body><h2>Executive Summary</h2></body></html>"
        wrapped = MeetingNotesApp._wrap_html_preview(full_doc)
        assert wrapped.startswith("<!DOCTYPE html>")
        assert '<article class="meeting-notes-article"><h2>Executive Summary</h2></article>' in wrapped
        assert wrapped.endswith("</body></html>")

        already_wrapped = "<!DOCTYPE html><html><body><article>Hi</article></body></html>"
        assert MeetingNotesApp._wrap_html_preview(already_wrapped) == already_wrapped

        fragment = "<p>Plain fragment</p>"
        assert MeetingNotesApp._wrap_html_preview(fragment) == '<article class="meeting-notes-article"><p>Plain fragment</p></article>'
        assert MeetingNotesApp._wrap_html_preview("") == ""

