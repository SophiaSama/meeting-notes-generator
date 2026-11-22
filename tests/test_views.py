"""Unit tests for pure presentation helpers in meeting_notes/views.py."""

from __future__ import annotations

import html
import re
import pytest

from meeting_notes.views import (
    STARTER_QUESTIONS,
    classify_error,
    format_stage_progress,
    map_progress_to_stage,
    render_empty_state,
    render_error_banner,
    ResultsPaneState,
    begin_generation,
    complete_generation,
    fail_generation,
    format_elapsed_time,
    render_provider_chip,
    reset_results_pane,
)


class TestRenderProviderChip:
    """Test suite for render_provider_chip presentation helper."""

    def test_configured_standard_model(self):
        """Standard model displays 'Connected: <model>' with connected CSS class."""
        rendered = render_provider_chip(is_configured=True, model="gpt-4o-mini")
        assert isinstance(rendered, str)
        assert "provider-chip" in rendered
        assert "connected" in rendered
        assert "unconfigured" not in rendered
        assert "Connected: gpt-4o-mini" in rendered
        assert "Not configured" not in rendered

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "claude-3-5-sonnet-20241022",
            "deepseek-reasoner",
            "meta-llama/Llama-3-70b-instruct",
            "custom:model-v1.0@prod",
        ],
    )
    def test_configured_various_models(self, model_name: str):
        """Various model name formats render correctly with proper escaping."""
        rendered = render_provider_chip(is_configured=True, model=model_name)
        assert f"Connected: {html.escape(model_name)}" in rendered
        assert "connected" in rendered
        assert "unconfigured" not in rendered

    def test_configured_empty_model(self):
        """When configured with empty model name, displays 'Connected' without dangling colon."""
        rendered = render_provider_chip(is_configured=True, model="")
        assert "Connected" in rendered
        assert "Connected:" not in rendered
        assert "connected" in rendered

    def test_configured_whitespace_model(self):
        """Whitespace in model name is stripped cleanly."""
        rendered = render_provider_chip(is_configured=True, model="  gpt-4o-mini  ")
        assert "Connected: gpt-4o-mini" in rendered

    def test_unconfigured_default(self):
        """Unconfigured state renders 'Not configured' with unconfigured CSS class."""
        rendered = render_provider_chip(is_configured=False)
        assert isinstance(rendered, str)
        assert "provider-chip" in rendered
        assert "unconfigured" in rendered
        assert "connected" not in rendered
        assert "Not configured" in rendered
        assert "Connected:" not in rendered

    def test_unconfigured_ignores_passed_model(self):
        """When is_configured is False, model name argument is ignored and not rendered."""
        rendered = render_provider_chip(is_configured=False, model="gpt-4o-mini")
        assert "Not configured" in rendered
        assert "gpt-4o-mini" not in rendered
        assert "Connected:" not in rendered
        assert "unconfigured" in rendered
        assert "connected" not in rendered

    @pytest.mark.parametrize(
        "malicious_model, expected_escaped_part",
        [
            ("<script>alert('xss')</script>", "&lt;script&gt;alert("),
            ('<img src="x" onerror="alert(1)">', "&lt;img src="),
            ('"><b onmouseover="alert(1)">hover</b>', "&quot;&gt;&lt;b "),
            ("model'\"&<>", "&amp;"),
        ],
    )
    def test_xss_escaping_prevention(self, malicious_model: str, expected_escaped_part: str):
        """Model names containing HTML tags or attributes are strictly escaped."""
        rendered = render_provider_chip(is_configured=True, model=malicious_model)
        assert "<script>" not in rendered
        assert "<img" not in rendered
        assert "<b>" not in rendered
        assert expected_escaped_part in rendered

    def test_accessibility_and_dom_elements(self):
        """Container has role='status' and aria-live='polite', with visual status dot."""
        connected_html = render_provider_chip(is_configured=True, model="gpt-4o")
        assert 'role="status"' in connected_html
        assert 'aria-live="polite"' in connected_html
        assert "provider-chip-dot" in connected_html
        assert 'aria-hidden="true"' in connected_html
        assert "provider-chip-text" in connected_html

        unconfigured_html = render_provider_chip(is_configured=False)
        assert 'role="status"' in unconfigured_html
        assert 'aria-live="polite"' in unconfigured_html
        assert "provider-chip-dot" in unconfigured_html
        assert "provider-chip-text" in unconfigured_html


class TestFoundationalViewsHelpers:
    """Tests for forward-looking view helpers (M3-M5) in views.py."""

    def test_render_empty_state_copy(self):
        html_out = render_empty_state()
        assert "Add a Recording or Transcript to begin" in html_out
        assert "empty-state-card" in html_out
        assert "No Meeting Notes yet" in html_out
        assert 'role="status"' in html_out

    def test_map_progress_to_stage_video_flow(self):
        assert map_progress_to_stage(0.05, "Reading video metadata...") == "Extracting Screenshots"
        assert map_progress_to_stage(0.20, "Extracting frames (30s) & transcribing...") == "Extracting Screenshots"
        assert map_progress_to_stage(0.40, "Transcribing audio chunk 2...") == "Transcribing"
        assert map_progress_to_stage(0.62, "Calling gpt-4o...") == "Summarizing"
        assert map_progress_to_stage(0.83, "Building HTML output...") == "Summarizing"
        assert map_progress_to_stage(1.0, "Done!") == "Done"
        assert map_progress_to_stage(0.99, "Done with processing") == "Done"

    def test_map_progress_to_stage_transcript_only_flow(self):
        assert map_progress_to_stage(0.10, "Reading uploaded transcript...", is_transcript_only=True) == "Summarizing"
        assert map_progress_to_stage(0.62, "Calling gpt-4o...", is_transcript_only=True) == "Summarizing"
        assert map_progress_to_stage(1.0, "Done!", is_transcript_only=True) == "Done"

    def test_format_stage_progress_under_one_minute(self):
        html_out = format_stage_progress("Summarizing", 25.4)
        assert "Summarizing" in html_out
        assert "(25s)" in html_out
        assert "stage-progress" in html_out
        assert 'role="status"' in html_out
        assert 'aria-live="polite"' in html_out
        assert "stage-spinner" in html_out

    def test_format_stage_progress_over_one_minute(self):
        html_out = format_stage_progress("Transcribing", 85.0)
        assert "Transcribing" in html_out
        assert "(1m 25s)" in html_out

    def test_format_stage_progress_done(self):
        html_out = format_stage_progress("Done", 120.0)
        assert "Done" in html_out
        assert "stage-done" in html_out
        assert "✓" in html_out
        assert "(2m 00s)" in html_out

    def test_classify_error_api_key(self):
        cause, fix = classify_error("Error 401: Invalid api key provided")
        assert "API key" in cause
        assert "Provider Settings" in fix

    def test_classify_error_ffmpeg(self):
        cause, fix = classify_error("FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'")
        assert "FFmpeg" in cause
        assert "PATH" in fix

    def test_classify_error_payload_size(self):
        cause, fix = classify_error("Error 413: payload too large / maximum context length exceeded")
        assert "limits" in cause or "exceeds" in cause
        assert "screenshot interval" in fix or "shorter" in fix

    def test_classify_error_empty(self):
        cause, fix = classify_error("")
        assert "unexpected" in cause.lower()
        assert "Provider Settings" in fix

    def test_classify_error_generic_fallback(self):
        cause, fix = classify_error("UnknownSocketTimeout: connection reset by peer")
        assert "UnknownSocketTimeout" in cause
        assert "logs" in fix

    def test_map_progress_to_stage_none_fraction(self):
        assert map_progress_to_stage(None, "Calling model...") == "Summarizing"
        assert map_progress_to_stage(None, "Done") == "Done"
        assert map_progress_to_stage(None, "Preparing video...") == "Extracting Screenshots"

    def test_map_progress_to_stage_none_or_empty_desc(self):
        assert map_progress_to_stage(0.20, "") == "Extracting Screenshots"
        assert map_progress_to_stage(0.50, None) == "Transcribing"
        assert map_progress_to_stage(0.75, "") == "Summarizing"

    def test_map_progress_to_stage_boundary_fractions(self):
        assert map_progress_to_stage(0.34, "Step") == "Extracting Screenshots"
        assert map_progress_to_stage(0.35, "Step") == "Transcribing"
        assert map_progress_to_stage(0.59, "Step") == "Transcribing"
        assert map_progress_to_stage(0.60, "Step") == "Summarizing"
        assert map_progress_to_stage(0.99, "Step") == "Summarizing"
        assert map_progress_to_stage(1.00, "Step") == "Done"

    def test_format_stage_progress_clamping_negative_time(self):
        html_out = format_stage_progress("Extracting Screenshots", -5.0)
        assert "(0s)" in html_out

    def test_format_stage_progress_html_escaping(self):
        script_stage = "<script>alert('stage')</script>"
        html_out = format_stage_progress(script_stage, 10.0)
        assert "<script>" not in html_out
        assert html.escape(script_stage) in html_out

    def test_classify_error_prefix_stripping(self):
        raw = "Generation failed. You can upload again and retry.\n\nError: RuntimeError: 401 Unauthorized"
        cause, fix = classify_error(raw)
        assert "API key" in cause
        assert "Provider Settings" in fix

    @pytest.mark.parametrize(
        "err_msg",
        [
            "401 Unauthorized",
            "API key invalid",
            "Bearer token missing",
            "403 Forbidden",
            "authentication failure",
        ],
    )
    def test_classify_error_auth_variations(self, err_msg: str):
        cause, fix = classify_error(err_msg)
        assert "API key" in cause
        assert "Provider Settings" in fix

    @pytest.mark.parametrize(
        "err_msg",
        [
            "FileNotFoundError: 'ffmpeg'",
            "ffprobe: command not found",
            "codec not recognized",
            "executable not found on PATH",
        ],
    )
    def test_classify_error_ffmpeg_variations(self, err_msg: str):
        cause, fix = classify_error(err_msg)
        assert "FFmpeg" in cause
        assert "PATH" in fix

    @pytest.mark.parametrize(
        "err_msg",
        [
            "413 Payload Too Large",
            "maximum context length exceeded",
            "context_length limit reached",
            "request entity too large",
            "payload exceeds limit",
        ],
    )
    def test_classify_error_payload_variations(self, err_msg: str):
        cause, fix = classify_error(err_msg)
        assert "limits" in cause or "exceeds" in cause
        assert "screenshot interval" in fix or "shorter" in fix

    def test_classify_error_missing_inputs(self):
        cause, fix = classify_error("Please upload a video file or a transcript file first.")
        assert "Missing Recording or Transcript" in cause
        assert "Add a Recording or Transcript" in fix

    def test_render_error_banner(self):
        html_out = render_error_banner("Invalid key <tag>", "Check settings & accounts")
        assert "error-banner" in html_out
        assert 'role="alert"' in html_out
        assert "Invalid key &lt;tag&gt;" in html_out
        assert "Check settings &amp; accounts" in html_out
        assert "onclick" in html_out
        assert "error-banner-dismiss" in html_out
        assert 'aria-label="Dismiss error"' in html_out

    def test_render_error_banner_xss_prevention(self):
        malicious_cause = "<script>alert('xss')</script>"
        malicious_fix = "<img src=x onerror=alert(1)>"
        html_out = render_error_banner(
            cause=malicious_cause,
            fix=malicious_fix,
        )
        assert "<script>" not in html_out
        assert "<img" not in html_out
        assert html.escape(malicious_cause) in html_out
        assert html.escape(malicious_fix) in html_out

    def test_starter_questions(self):
        assert isinstance(STARTER_QUESTIONS, list)
        assert len(STARTER_QUESTIONS) == 2
        assert STARTER_QUESTIONS[0] == "What were the action items?"
        assert STARTER_QUESTIONS[1] == "Who owns what?"
        for q in STARTER_QUESTIONS:
            assert isinstance(q, str)
            assert len(q) > 0


class TestResultsPaneStateMachine:
    """Interface-level tests for the deepened ResultsPaneState transitions."""

    def test_begin_generation_sets_generating_phase(self):
        state = begin_generation(is_transcript_only=False)
        assert isinstance(state, ResultsPaneState)
        assert state.phase == "generating"
        assert state.empty_visible is False
        assert state.results_visible is False
        assert state.error_visible is False
        assert state.starter_visible is False
        assert state.stage_visible is True
        assert "Extracting Screenshots" in state.stage_html

        tx_state = begin_generation(is_transcript_only=True)
        assert tx_state.stage_visible is True
        assert "Summarizing" in tx_state.stage_html

    def test_complete_generation_sets_populated_phase(self):
        ctx = {"markdown_notes": "# Notes"}
        state = complete_generation(["t.png"], "# Notes", "<p>Notes</p>", "out.html", ctx)
        assert state.phase == "populated"
        assert state.empty_visible is False
        assert state.results_visible is True
        assert state.error_visible is False
        assert state.error_html == ""
        assert state.starter_visible is True
        assert state.stage_visible is False
        assert state.stage_html == ""
        assert state.meeting_context is ctx

    def test_fail_generation_sets_error_phase_with_actionable_banner(self):
        state = fail_generation("Error: 401 Unauthorized")
        assert state.phase == "error"
        assert state.empty_visible is True
        assert state.results_visible is False
        assert state.error_visible is True
        assert "Invalid or missing API key." in state.error_html
        assert state.starter_visible is False
        assert state.stage_visible is False
        assert state.meeting_context is None

    def test_reset_results_pane_restores_empty_phase(self):
        state = reset_results_pane()
        assert state.phase == "empty"
        assert state.empty_visible is True
        assert state.results_visible is False
        assert state.error_visible is False
        assert state.error_html == ""
        assert state.starter_visible is False
        assert state.stage_visible is False
        assert state.stage_html == ""
        assert state.thumbnails == []
        assert state.meeting_context is None

    def test_format_elapsed_time(self):
        assert format_elapsed_time(0.0) == "0s"
        assert format_elapsed_time(59.4) == "59s"
        assert format_elapsed_time(65.0) == "1m 05s"

    def test_screenshots_visible_across_all_phases(self):
        assert reset_results_pane().screenshots_visible is False
        assert begin_generation(is_transcript_only=False).screenshots_visible is False

        tx_state = begin_generation(is_transcript_only=True)
        assert tx_state.screenshots_visible is False
        assert 'data-stage="Extracting Screenshots" data-state="skipped"' in tx_state.stage_html
        assert 'data-stage="Transcribing" data-state="skipped"' in tx_state.stage_html

        pop_with_shots = complete_generation(
            ["shot1.png"], "# Notes", "<p>Notes</p>", "out.html", {"k": "v"}
        )
        assert pop_with_shots.screenshots_visible is True

        pop_no_shots = complete_generation(
            [], "# Notes", "<p>Notes</p>", "out.html", {"k": "v"}
        )
        assert pop_no_shots.screenshots_visible is False

        assert fail_generation("Error: 401 Unauthorized").screenshots_visible is False


def _extract_step_li(html_out: str, stage_name: str) -> str:
    pattern = rf'<li\b[^>]*data-stage="{re.escape(stage_name)}"[^>]*>.*?</li>'
    match = re.search(pattern, html_out, flags=re.DOTALL)
    assert match is not None, f"Could not find step <li> for {stage_name!r} in:\n{html_out}"
    return match.group(0)


class TestFourStepProgressCard:
    """Unit tests for the 4-step progress card rendered by format_stage_progress."""

    def test_all_four_canonical_stages_rendered(self):
        html_out = format_stage_progress("Extracting Screenshots", 0.0)
        assert "stage-progress-card" in html_out
        assert "stage-steps" in html_out
        for stage in ("Extracting Screenshots", "Transcribing", "Summarizing", "Done"):
            assert f'data-stage="{stage}"' in html_out
            assert stage in html_out

    def test_step_states_extracting_active(self):
        html_out = format_stage_progress("Extracting Screenshots", 4.0)
        assert "stage-spinner" in html_out
        assert "(4s)" in html_out
        assert "stage-done" not in html_out
        assert "✓" not in html_out

        s1 = _extract_step_li(html_out, "Extracting Screenshots")
        s2 = _extract_step_li(html_out, "Transcribing")
        s3 = _extract_step_li(html_out, "Summarizing")
        s4 = _extract_step_li(html_out, "Done")

        assert 'data-state="active"' in s1
        assert 'aria-current="step"' in s1
        assert 'data-state="pending"' in s2
        assert 'data-state="pending"' in s3
        assert 'data-state="pending"' in s4

    def test_step_states_transcribing_active(self):
        html_out = format_stage_progress("Transcribing", 35.0)
        assert "(35s)" in html_out
        assert "stage-done" not in html_out
        assert "✓" not in html_out

        s1 = _extract_step_li(html_out, "Extracting Screenshots")
        s2 = _extract_step_li(html_out, "Transcribing")
        s3 = _extract_step_li(html_out, "Summarizing")
        s4 = _extract_step_li(html_out, "Done")

        assert 'data-state="completed"' in s1
        assert "stage-step-completed" in s1
        assert "stage-step-check-icon" in s1
        assert 'data-state="active"' in s2
        assert 'aria-current="step"' in s2
        assert 'data-state="pending"' in s3
        assert 'data-state="pending"' in s4

    def test_step_states_summarizing_active(self):
        html_out = format_stage_progress("Summarizing", 90.0)
        assert "(1m 30s)" in html_out
        assert "stage-done" not in html_out
        assert "✓" not in html_out

        s1 = _extract_step_li(html_out, "Extracting Screenshots")
        s2 = _extract_step_li(html_out, "Transcribing")
        s3 = _extract_step_li(html_out, "Summarizing")
        s4 = _extract_step_li(html_out, "Done")

        assert 'data-state="completed"' in s1
        assert 'data-state="completed"' in s2
        assert 'data-state="active"' in s3
        assert 'aria-current="step"' in s3
        assert 'data-state="pending"' in s4

    def test_step_states_done(self):
        html_out = format_stage_progress("Done", 125.0)
        assert "stage-done" in html_out
        assert "stage-check" in html_out
        assert "✓" in html_out
        assert "(2m 05s)" in html_out
        assert "stage-spinner" not in html_out

        for stage in ("Extracting Screenshots", "Transcribing", "Summarizing", "Done"):
            step_li = _extract_step_li(html_out, stage)
            assert 'data-state="completed"' in step_li
            assert "stage-step-check-icon" in step_li
            assert 'aria-current="step"' not in step_li

    def test_transcript_only_marks_video_stages_skipped(self):
        active_html = format_stage_progress("Summarizing", 12.0, transcript_only=True)
        assert "(12s)" in active_html
        assert "✓" not in active_html
        assert "stage-done" not in active_html

        s1 = _extract_step_li(active_html, "Extracting Screenshots")
        s2 = _extract_step_li(active_html, "Transcribing")
        s3 = _extract_step_li(active_html, "Summarizing")
        s4 = _extract_step_li(active_html, "Done")

        assert 'data-state="skipped"' in s1
        assert "Skipped" in s1
        assert 'data-state="skipped"' in s2
        assert "Skipped" in s2
        assert 'data-state="active"' in s3
        assert 'data-state="pending"' in s4

        done_html = format_stage_progress("Done", 20.0, transcript_only=True)
        assert "stage-done" in done_html
        assert "✓" in done_html
        assert 'data-state="skipped"' in _extract_step_li(done_html, "Extracting Screenshots")
        assert 'data-state="skipped"' in _extract_step_li(done_html, "Transcribing")
        assert 'data-state="completed"' in _extract_step_li(done_html, "Summarizing")
        assert 'data-state="completed"' in _extract_step_li(done_html, "Done")
