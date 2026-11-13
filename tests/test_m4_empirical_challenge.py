"""Empirical challenge tests for Milestone 4 (StageProgressTracker, timing, transcript-only skipping, and error banner).

This module contains stress-test harnesses, property-based verification, and boundary condition checks
designed by Milestone 4 Challenger 1.
"""

from __future__ import annotations

import html
import itertools
from unittest.mock import MagicMock, patch

import pytest

from meeting_notes.ui import MeetingNotesApp, StageProgressTracker
from meeting_notes.views import (
    classify_error,
    format_stage_progress,
    map_progress_to_stage,
    render_error_banner,
)


class TestStageProgressTrackerAdversarial:
    """Stress-test StageProgressTracker with out-of-order, indeterminate, and boundary inputs."""

    def test_out_of_order_progress_fractions(self):
        """Pipeline emitting out-of-order fractions updates stage and history without crashing."""
        mock_fn = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_fn)

        # Non-monotonic sequence: 0.8 -> 0.2 -> 0.5 -> 0.1 -> 1.0
        fractions = [0.8, 0.2, 0.5, 0.1, 1.0]
        expected_stages = ["Summarizing", "Extracting Screenshots", "Transcribing", "Extracting Screenshots", "Done"]

        for frac, expected in zip(fractions, expected_stages):
            tracker(frac)
            assert tracker.current_stage == expected
            assert tracker.last_fraction == frac

        assert len(tracker.history) == len(fractions)
        assert [entry["stage"] for entry in tracker.history] == expected_stages

    def test_indeterminate_fraction_with_various_descriptions(self):
        """Indeterminate fraction (None) delegates properly and uses description or defaults."""
        mock_fn = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_fn)

        tracker(None, desc="Starting worker thread...")
        assert tracker.current_stage == "Extracting Screenshots"
        assert tracker.last_fraction is None
        mock_fn.assert_called_with(None, desc=tracker.history[-1]["formatted_desc"])

        tracker(None, desc="Calling model...")
        assert tracker.current_stage == "Summarizing"

        tracker(None, desc="Done with all stages")
        assert tracker.current_stage == "Done"

    def test_special_numeric_fractions(self):
        """Special numeric values (nan, inf, negative, >1.0) are handled safely."""
        mock_fn = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_fn)

        # Negative fraction (<0.35)
        tracker(-0.5, desc="Step")
        assert tracker.current_stage == "Extracting Screenshots"

        # Fraction > 1.0 (>=1.0 -> Done)
        tracker(2.5, desc="Step")
        assert tracker.current_stage == "Done"

        # Infinity (>=1.0 -> Done)
        tracker(float("inf"), desc="Step")
        assert tracker.current_stage == "Done"

        # NaN does not satisfy >= 1.0 or < 0.35 or < 0.60; falls through cleanly to Summarizing
        tracker(float("nan"), desc="Step")
        assert tracker.current_stage == "Summarizing"

    def test_non_string_descriptions(self):
        """Verify behavior with None and non-string descriptions."""
        mock_fn = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_fn)

        # None is safely converted to empty string
        tracker(0.25, desc=None)
        assert tracker.current_stage == "Extracting Screenshots"
        assert tracker.last_desc is None

        # Empty string
        tracker(0.50, desc="")
        assert tracker.current_stage == "Transcribing"

        # Explicit non-string desc (e.g. integer) raises AttributeError on .lower()
        # This is expected behavior given the type contract desc: str | None
        with pytest.raises(AttributeError):
            tracker(0.50, desc=123)

    def test_extra_kwargs_forwarded_to_progress_callable(self):
        """Additional kwargs such as total or unit are forwarded to underlying progress function."""
        mock_fn = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_fn)

        tracker(0.40, desc="Transcribing chunk...", total=10, unit="chunk")
        mock_fn.assert_called_with(
            0.40,
            desc=tracker.history[-1]["formatted_desc"],
            total=10,
            unit="chunk",
        )

    def test_progress_callable_type_error_fallback(self):
        """If progress callable does not accept desc kwarg, tracker falls back to positional call."""
        calls = []

        def simple_progress(fraction):
            calls.append(fraction)

        tracker = StageProgressTracker(progress_fn=simple_progress)
        tracker(0.65, desc="Summarizing...")
        assert calls == [0.65]
        assert tracker.current_stage == "Summarizing"


class TestElapsedTimeBoundaryDurations:
    """Stress-test elapsed time formatting across exact boundary durations."""

    @pytest.mark.parametrize(
        "elapsed, expected_substr",
        [
            (0.0, "(0s)"),
            (0.4, "(0s)"),
            (1.0, "(1s)"),
            (59.0, "(59s)"),
            (59.4, "(59s)"),
            (60.0, "(1m 00s)"),
            (61.0, "(1m 01s)"),
            (119.0, "(1m 59s)"),
            (120.0, "(2m 00s)"),
            (3599.0, "(59m 59s)"),
            (3600.0, "(60m 00s)"),
            (7265.0, "(121m 05s)"),
            (-1.0, "(0s)"),
            (-50.0, "(0s)"),
        ],
    )
    def test_format_stage_progress_boundaries(self, elapsed: float, expected_substr: str):
        rendered = format_stage_progress("Transcribing", elapsed)
        assert expected_substr in rendered
        assert 'role="status"' in rendered
        assert 'aria-live="polite"' in rendered

    def test_format_stage_progress_done_vs_active_styling(self):
        active_html = format_stage_progress("Summarizing", 15.0)
        assert "stage-spinner" in active_html
        assert "stage-done" not in active_html
        assert "✓" not in active_html

        done_html = format_stage_progress("Done", 45.0)
        assert "stage-done" in done_html
        assert "✓" in done_html
        assert "stage-check" in done_html

    def test_stage_progress_tracker_mock_time_elapsed(self):
        """StageProgressTracker formats elapsed time using start_time offset."""
        mock_prog = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_prog)
        base_time = 5000.0
        tracker.start_time = base_time

        boundaries = [
            (0.0, "Extracting Screenshots (0s)"),
            (59.0, "Transcribing (59s)"),
            (60.0, "Transcribing (1m 00s)"),
            (3600.0, "Summarizing (60m 00s)"),
        ]

        for offset, expected in boundaries:
            with patch("time.monotonic", return_value=base_time + offset):
                frac = 0.1 if offset == 0 else (0.5 if offset in (59, 60) else 0.8)
                tracker(frac)
                assert mock_prog.call_args[1]["desc"] == expected


class TestTranscriptOnlySkippingProperty:
    """Property-based verification that transcript-only runs bypass video stages."""

    def test_exhaustive_cartesian_transcript_only_invariants(self):
        """For any fraction and description, transcript-only runs NEVER produce video stages."""
        test_fractions = [None, -0.5, 0.0, 0.05, 0.20, 0.34, 0.35, 0.50, 0.59, 0.60, 0.80, 1.0, 2.0]
        test_descs = [
            "",
            None,
            "Extracting frames (30s)...",
            "Extracting screenshots...",
            "Reading video metadata...",
            "Preparing video...",
            "Transcribing audio chunk 1...",
            "Calling model...",
            "Summarizing...",
            "Done!",
            "Random pipeline event",
        ]

        forbidden_stages = {"Extracting Screenshots", "Transcribing"}

        for frac, desc in itertools.product(test_fractions, test_descs):
            stage = map_progress_to_stage(frac, desc, is_transcript_only=True)
            assert stage in {"Summarizing", "Done"}, f"Unexpected stage '{stage}' for fraction={frac}, desc={desc}"
            assert stage not in forbidden_stages, f"Forbidden stage '{stage}' leaked in transcript-only mode!"

    def test_simulated_transcript_only_run_lifecycle(self):
        """Simulate a complete transcript-only run through StageProgressTracker."""
        mock_prog = MagicMock()
        tracker = StageProgressTracker(progress_fn=mock_prog, is_transcript_only=True)

        pipeline_events = [
            (0.10, "Reading uploaded transcript..."),
            (0.57, "Condensing long transcript in segments..."),
            (0.62, "Calling gpt-4o..."),
            (0.83, "Building HTML output..."),
            (1.0, "Done!"),
        ]

        for frac, desc in pipeline_events:
            tracker(frac, desc=desc)

        stages_visited = [entry["stage"] for entry in tracker.history]
        assert "Extracting Screenshots" not in stages_visited
        assert "Transcribing" not in stages_visited
        assert stages_visited == ["Summarizing", "Summarizing", "Summarizing", "Summarizing", "Done"]


class TestErrorBannerActionableAdversarial:
    """Adversarial testing of Ticket 05 error banner classification, escaping, and lifecycle."""

    @pytest.mark.parametrize(
        "error_str, expected_cause_substr, expected_fix_substr",
        [
            (
                "openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}",
                "API key",
                "Provider Settings",
            ),
            (
                "HTTPError: 403 Forbidden: Account credits exhausted or access denied",
                "API key",
                "Provider Settings",
            ),
            (
                "subprocess.CalledProcessError: Command '['ffmpeg', '-i', ...] returned non-zero exit status 127",
                "FFmpeg",
                "PATH",
            ),
            (
                "[WinError 2] The system cannot find the file specified: 'ffmpeg.exe'",
                "FFmpeg",
                "PATH",
            ),
            (
                "BadRequestError: Error code: 413 - {'error': {'message': 'Request entity too large, max_tokens exceeded'}}",
                "limits",
                "screenshot interval",
            ),
            (
                "ContextWindowExceededError: Model maximum context length is 128000 tokens",
                "limits",
                "screenshot interval",
            ),
            (
                "Please upload a video file or a transcript file first.",
                "Missing Recording or Transcript",
                "Add a Recording or Transcript",
            ),
        ],
    )
    def test_classify_error_known_patterns(self, error_str: str, expected_cause_substr: str, expected_fix_substr: str):
        cause, fix = classify_error(error_str)
        assert expected_cause_substr.lower() in cause.lower()
        assert expected_fix_substr.lower() in fix.lower()

    def test_classify_error_long_message_truncation(self):
        """Very long unknown error strings are truncated cleanly to prevent UI blowout."""
        long_err = "InternalSystemFailure: " + ("x" * 2000)
        cause, fix = classify_error(long_err)
        assert len(cause) <= 265  # 'Processing failed: ' + 240 chars
        assert "Check application logs" in fix

    def test_render_error_banner_xss_escaping(self):
        """Malicious code in error cause and fix is HTML-escaped."""
        xss_cause = "<script>alert('pwned')</script>"
        xss_fix = "<img src='x' onerror='alert(2)'>"
        rendered = render_error_banner(xss_cause, xss_fix)

        assert "<script>" not in rendered
        assert "<img" not in rendered
        assert html.escape(xss_cause) in rendered
        assert html.escape(xss_fix) in rendered
        assert 'role="alert"' in rendered
        assert "onclick=" in rendered
        assert 'aria-label="Dismiss error"' in rendered

    def test_full_error_lifecycle_generate_and_reset(self):
        """Error -> banner visible -> reset -> banner cleared -> success -> banner cleared."""
        app = MeetingNotesApp()

        # Step 1: Trigger failure
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline_safe.return_value = (
            [],
            "Generation failed. Error: 401 Unauthorized",
            "",
            None,
            None,
        )
        mock_pipeline._is_valid_meeting_context.return_value = False
        app.pipeline_service = mock_pipeline

        err_out = app._on_generate_click(
            "test.mp4", "https://api.openai.com", "key", "model", 30, 8, "Summary", "English"
        )
        assert len(err_out) == 12
        assert err_out[7]["visible"] is True  # empty_state visible
        assert err_out[8]["visible"] is False  # results hidden
        assert err_out[9]["visible"] is True  # error_banner visible
        assert "Invalid or missing API key." in err_out[9]["value"]
        assert err_out[10]["visible"] is False  # starter_row hidden

        # Step 2: Reset clears error banner
        reset_out = app._on_reset()
        assert len(reset_out) == 16
        assert reset_out[13]["visible"] is False
        assert reset_out[13]["value"] == ""
        assert reset_out[14]["visible"] is False  # starter_row hidden

        # Step 3: Success leaves error banner hidden
        mock_pipeline.run_pipeline_safe.return_value = (
            ["thumb.png"],
            "Meeting Notes content",
            "<p>Notes</p>",
            "out.html",
            {"meeting_notes": "content"},
        )
        mock_pipeline._is_valid_meeting_context.return_value = True

        succ_out = app._on_generate_click(
            "test.mp4", "https://api.openai.com", "key", "model", 30, 8, "Summary", "English"
        )
        assert len(succ_out) == 12
        assert succ_out[7]["visible"] is False  # empty_state hidden
        assert succ_out[8]["visible"] is True   # results visible
        assert succ_out[9]["visible"] is False  # error_banner hidden
        assert succ_out[9]["value"] == ""
        assert succ_out[10]["visible"] is True  # starter_row visible
