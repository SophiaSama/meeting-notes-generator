"""Unit and seam tests for MeetingNotesApp handler functions and UI event graph wiring."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from meeting_notes.ui import MeetingNotesApp, StageProgressTracker
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
        assert len(start_fn.outputs) == 4
        assert app.stage_progress_html._id in [c._id for c in start_fn.outputs]

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
        """_on_generate_click returns 10 outputs, hiding empty state, revealing results container, and hiding error banner."""
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
        """_on_reset returns 15 outputs, restoring empty state to visible, hiding results container, error banner, and starter row."""
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
        assert app.error_banner.visible is False
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
        """_on_generate_start hides empty/results/error and shows stage_progress_html."""
        app = MeetingNotesApp()
        empty_u, results_u, err_u, stage_u = app._on_generate_start(video_file="meeting.mp4")
        assert empty_u.get("visible") is False
        assert results_u.get("visible") is False
        assert err_u.get("visible") is False
        assert stage_u.get("visible") is True
        assert "Extracting Screenshots" in stage_u.get("value", "")

        _, _, _, stage_tx = app._on_generate_start(video_file=None, transcript_file="notes.txt")
        assert stage_tx.get("visible") is True
        assert "Summarizing" in stage_tx.get("value", "")
