"""Playwright E2E tests verifying Light and Dark themes across Meeting Notes Helper.

Covers:
- System ``prefers-color-scheme`` initialization (light and dark)
- Interactive theme toggle button label, ``aria-pressed``, ``aria-label``, and ``colorScheme``
- ``localStorage['meeting-notes-theme']`` persistence across page reloads (overriding OS preference)
- Cross-tab synchronization via the ``storage`` event
- Computed visual styling and luminance contrast in both Light and Dark themes for:
  - Page container & body
  - Provider status chip (``.provider-chip.connected``)
  - Empty state card (``.empty-state-card``)
  - Capture hint box (``.capture-hint-box``)
  - Error banner (``.error-banner``)
  - Rendered Meeting Notes (``.rendered-notes-html``)
  - Screenshots gallery dark-mode background (no stark white box)
  - Meeting Chat starter chips (``.chat-starter-btn``)
"""

from __future__ import annotations

import re
import socket
from pathlib import Path
from typing import Any

import gradio as gr
import pytest
from PIL import Image

from meeting_notes.bootstrap import ProxyDefaults
from meeting_notes.notes import NotesService
from meeting_notes.pipeline import MeetingContext, PipelineService
from meeting_notes.ui import CSS, THEME, THEME_HEAD_HTML, MeetingNotesApp
from meeting_notes.video import VideoService
from tests.conftest import FakeLLMService

pytestmark = [pytest.mark.playwright, pytest.mark.e2e]

playwright_sync = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright is required to run browser E2E tests",
)
sync_playwright = playwright_sync.sync_playwright


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_browser(p: Any) -> Any:
    """Launch headless Chromium, falling back to installed Chrome if needed."""
    try:
        return p.chromium.launch(headless=True)
    except Exception:
        return p.chromium.launch(headless=True, channel="chrome")


def _parse_rgb(color: str) -> tuple[int, int, int, float]:
    """Parse ``rgb(r, g, b)`` or ``rgba(r, g, b, a)`` into ``(r, g, b, alpha)``."""
    match = re.match(
        r"rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(?:\s*,\s*(\d+(?:\.\d+)?))?\s*\)",
        color.strip(),
    )
    if not match:
        raise AssertionError(f"Unexpected CSS color format: {color!r}")
    r, g, b = (int(float(match.group(i))) for i in (1, 2, 3))
    alpha = float(match.group(4)) if match.group(4) is not None else 1.0
    return r, g, b, alpha


def _luminance(color: str) -> float:
    """Return perceptual luminance in ``[0.0, 1.0]`` for an opaque RGB color."""
    r, g, b, _ = _parse_rgb(color)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _effective_bg(page: Any, selector: str) -> str:
    """Walk up ancestors from ``selector`` until a non-transparent background is found."""
    return page.evaluate(
        """(sel) => {
            let el = document.querySelector(sel);
            while (el) {
                const bg = getComputedStyle(el).backgroundColor;
                if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                    return bg;
                }
                el = el.parentElement;
            }
            return getComputedStyle(document.body).backgroundColor;
        }""",
        selector,
    )


def _computed_style(page: Any, selector: str, prop: str) -> str:
    return page.evaluate(
        """([sel, prop]) => {
            const el = document.querySelector(sel);
            if (!el) return '';
            return getComputedStyle(el)[prop] || '';
        }""",
        [selector, prop],
    )


def _wait_for_theme(page: Any, expected_theme: str) -> None:
    """Assert and wait until DOM classes, colorScheme, and toggle button match ``expected_theme``."""
    expect_dark = expected_theme == "dark"
    expected_label = "🌙 Dark" if expect_dark else "☀️ Light"
    expected_aria_label = (
        "Theme: Dark (switch to Light)"
        if expect_dark
        else "Theme: Light (switch to Dark)"
    )

    page.wait_for_function(
        """([expectDark, expectedLabel]) => {
            const rootDark = document.documentElement.classList.contains('dark');
            const bodyDark = document.body && document.body.classList.contains('dark');
            const btn = document.querySelector('#theme-toggle-btn');
            const text = btn ? btn.textContent.trim() : '';
            return rootDark === expectDark && bodyDark === expectDark && text === expectedLabel;
        }""",
        arg=[expect_dark, expected_label],
        timeout=10_000,
    )

    state = page.evaluate(
        """() => {
            const btn = document.querySelector('#theme-toggle-btn');
            return {
                rootDark: document.documentElement.classList.contains('dark'),
                bodyDark: Boolean(document.body && document.body.classList.contains('dark')),
                colorScheme: document.documentElement.style.colorScheme,
                text: btn ? btn.textContent.trim() : '',
                ariaPressed: btn ? btn.getAttribute('aria-pressed') : null,
                ariaLabel: btn ? btn.getAttribute('aria-label') : null,
            };
        }"""
    )
    assert state["rootDark"] is expect_dark
    assert state["bodyDark"] is expect_dark
    assert state["colorScheme"] == expected_theme
    assert state["text"] == expected_label
    assert state["ariaPressed"] == ("true" if expect_dark else "false")
    assert state["ariaLabel"] == expected_aria_label
    # Allow CSS color transitions (e.g., 0.2s on .provider-chip) to settle
    page.wait_for_timeout(300)


class _ThemeTestVideoService(VideoService):
    """Minimal video stub returning deterministic metadata without touching ffmpeg/cv2."""

    def __init__(self):
        pass

    def get_video_info(self, video_path: str) -> dict[str, Any]:
        return {"duration": 120.0, "resolution": "1280x720", "fps": 30.0}

    def sample_frame_bytes(self, *args: Any, **kwargs: Any) -> list[int]:
        return []


class _ThemeTestPipelineService(PipelineService):
    """Fast deterministic pipeline stub for browser theme verification."""

    def __init__(self, screenshot_paths: list[str], html_export_path: str):
        self._screenshot_paths = screenshot_paths
        self._html_export_path = html_export_path

    def check_upload_for_duplicate(self, video_path: str | None) -> dict[str, Any]:
        return gr.update(value="", visible=False)

    def run_pipeline_safe(
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
        progress=None,
        return_context=False,
    ):
        if video_file is None and transcript_file is None:
            err = "Error: Please upload a Recording or Transcript file first."
            if return_context:
                return [], err, "", None, None
            return [], err, "", None

        notes_md = (
            "## Executive Summary\n\n"
            "- Reviewed Q4 roadmap and confirmed theme polish.\n"
            "- Verified contrast across Light and Dark modes.\n"
        )
        html_notes = (
            "<article>"
            "<h2>Executive Summary</h2>"
            "<p>Reviewed Q4 roadmap and confirmed theme polish.</p>"
            "<ul><li><strong>Alice</strong>: Verify Light and Dark contrast.</li></ul>"
            "<p class='meta'>Generated by test-theme-model</p>"
            "</article>"
        )
        thumbnails = [
            (self._screenshot_paths[0], "Screenshot 1 (0:40)"),
            (self._screenshot_paths[1], "Screenshot 2 (1:20)"),
        ]
        ctx = MeetingContext(
            markdown_notes=notes_md,
            transcript_evidence="[Chunk 1]\nAlice: Let us verify both themes.\n",
            meeting_title="Theme Verification Sync",
            source_type="video" if video_file else "transcript",
            model=model or "test-theme-model",
            output_language=output_language or "English",
        )
        if return_context:
            return thumbnails, notes_md, html_notes, self._html_export_path, ctx
        return thumbnails, notes_md, html_notes, self._html_export_path


@pytest.fixture(scope="module")
def theme_e2e_assets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Create temporary screenshot PNGs, HTML export, and a dummy Recording file."""
    root: Path = tmp_path_factory.mktemp("theme_e2e")
    shot1 = root / "shot1.png"
    shot2 = root / "shot2.png"
    Image.new("RGB", (320, 180), color=(40, 80, 140)).save(shot1)
    Image.new("RGB", (320, 180), color=(90, 50, 120)).save(shot2)

    html_export = root / "meeting_notes.html"
    html_export.write_text("<html><body><h1>Meeting Notes</h1></body></html>", encoding="utf-8")

    dummy_recording = root / "sample_meeting.mp4"
    dummy_recording.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    return {
        "screenshots": [str(shot1), str(shot2)],
        "html_export": str(html_export),
        "recording": str(dummy_recording),
    }


@pytest.fixture(scope="module")
def live_server_url(theme_e2e_assets: dict[str, Any]) -> str:
    """Launch a live Gradio instance backed by fast fake services on a free port."""
    defaults = ProxyDefaults(
        base_url="http://fake-provider.local/v1",
        api_key="sk-theme-e2e-test-key",
        model="test-theme-model",
        reasoning_effort="none",
    )
    mp = pytest.MonkeyPatch()
    mp.setattr(ProxyDefaults, "from_env", classmethod(lambda cls: defaults))
    try:
        app = MeetingNotesApp(
            llm_service=FakeLLMService(reply="ok"),
            video_service=_ThemeTestVideoService(),
            notes_service=NotesService(),
            pipeline_service=_ThemeTestPipelineService(
                screenshot_paths=theme_e2e_assets["screenshots"],
                html_export_path=theme_e2e_assets["html_export"],
            ),
        )
    finally:
        mp.undo()
    port = _find_free_port()
    app.demo.launch(
        css=CSS,
        theme=THEME,
        head=THEME_HEAD_HTML,
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        quiet=True,
        show_error=True,
        prevent_thread_lock=True,
    )
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        app.demo.close()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = _launch_browser(p)
        try:
            yield b
        finally:
            b.close()


@pytest.mark.parametrize("os_scheme", ["light", "dark"])
def test_system_color_scheme_initial_theme(browser, live_server_url: str, os_scheme: str):
    """When localStorage has no saved theme, the UI follows prefers-color-scheme."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        color_scheme=os_scheme,
    )
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_server_url, wait_until="domcontentloaded")
        _wait_for_theme(page, os_scheme)
        saved = page.evaluate("() => localStorage.getItem('meeting-notes-theme')")
        assert saved is None
        assert errors == []
    finally:
        context.close()


def test_theme_toggle_and_localstorage_persistence(browser, live_server_url: str):
    """Clicking #theme-toggle-btn toggles Light <-> Dark and persists across reloads."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        color_scheme="light",
    )
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_server_url, wait_until="domcontentloaded")
        _wait_for_theme(page, "light")

        # Toggle Light -> Dark
        page.locator("#theme-toggle-btn").click()
        _wait_for_theme(page, "dark")
        assert page.evaluate("() => localStorage.getItem('meeting-notes-theme')") == "dark"

        # Reload: saved Dark theme must persist even though OS color_scheme is 'light'
        page.reload(wait_until="domcontentloaded")
        _wait_for_theme(page, "dark")
        assert page.evaluate("() => localStorage.getItem('meeting-notes-theme')") == "dark"

        # Toggle Dark -> Light
        page.locator("#theme-toggle-btn").click()
        _wait_for_theme(page, "light")
        assert page.evaluate("() => localStorage.getItem('meeting-notes-theme')") == "light"

        # Reload: saved Light theme must persist
        page.reload(wait_until="domcontentloaded")
        _wait_for_theme(page, "light")
        assert errors == []
    finally:
        context.close()


def test_saved_light_theme_overrides_os_dark_preference(browser, live_server_url: str):
    """A saved 'light' preference in localStorage overrides OS dark mode on initial load."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        color_scheme="dark",
    )
    context.add_init_script("localStorage.setItem('meeting-notes-theme', 'light');")
    page = context.new_page()
    try:
        page.goto(live_server_url, wait_until="domcontentloaded")
        _wait_for_theme(page, "light")
    finally:
        context.close()


def test_cross_tab_theme_sync(browser, live_server_url: str):
    """Toggling the theme in one tab synchronizes another open tab via the storage event."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        color_scheme="light",
    )
    page1 = context.new_page()
    page2 = context.new_page()
    try:
        page1.goto(live_server_url, wait_until="domcontentloaded")
        page2.goto(live_server_url, wait_until="domcontentloaded")
        _wait_for_theme(page1, "light")
        _wait_for_theme(page2, "light")

        # Toggle in page1 -> page2 should sync to dark automatically
        page1.locator("#theme-toggle-btn").click()
        _wait_for_theme(page1, "dark")
        _wait_for_theme(page2, "dark")

        # Toggle back in page2 -> page1 should sync to light automatically
        page2.locator("#theme-toggle-btn").click()
        _wait_for_theme(page2, "light")
        _wait_for_theme(page1, "light")
    finally:
        context.close()


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_theme_visual_styling_initial_and_error_states(
    browser, live_server_url: str, theme_name: str
):
    """Verify visual styling and contrast of initial UI & error banner in each theme."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        color_scheme=theme_name,
    )
    context.add_init_script(f"localStorage.setItem('meeting-notes-theme', '{theme_name}');")
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_server_url, wait_until="domcontentloaded")
        _wait_for_theme(page, theme_name)

        # 1. Page background & body text luminance match the active theme
        body_bg = _effective_bg(page, ".gradio-container")
        body_fg = _computed_style(page, "body", "color")
        bg_lum = _luminance(body_bg)
        fg_lum = _luminance(body_fg)
        if theme_name == "dark":
            assert bg_lum < 0.25, f"Expected dark page background, got {body_bg}"
            assert fg_lum > 0.65, f"Expected light body text in dark mode, got {body_fg}"
        else:
            assert bg_lum > 0.75, f"Expected light page background, got {body_bg}"
            assert fg_lum < 0.35, f"Expected dark body text in light mode, got {body_fg}"

        # 2. Provider status chip is visible and readable
        chip = page.locator(".provider-chip.connected")
        chip.wait_for(state="visible", timeout=5_000)
        assert "Connected: test-theme-model" in chip.inner_text()
        chip_fg = _computed_style(page, ".provider-chip.connected", "color")
        if theme_name == "dark":
            assert _luminance(chip_fg) > 0.55, f"Expected light chip text in dark mode, got {chip_fg}"
        else:
            assert _luminance(chip_fg) < 0.45, f"Expected dark chip text in light mode, got {chip_fg}"

        # 3. Empty state card is visible with appropriate surface luminance
        empty_card = page.locator(".empty-state-card")
        empty_card.wait_for(state="visible", timeout=5_000)
        empty_bg = _effective_bg(page, ".empty-state-card")
        if theme_name == "dark":
            assert _luminance(empty_bg) < 0.30, f"Expected dark empty card bg, got {empty_bg}"
        else:
            assert _luminance(empty_bg) > 0.75, f"Expected light empty card bg, got {empty_bg}"

        # 4. Trigger error banner (Generate with no Recording/Transcript) and verify theme colors
        page.get_by_role("button", name="Generate Meeting Notes").click()
        banner = page.locator(".error-banner")
        banner.wait_for(state="visible", timeout=10_000)
        banner_bg = _effective_bg(page, ".error-banner")
        banner_fg = _computed_style(page, ".error-banner", "color")
        if theme_name == "dark":
            assert _luminance(banner_bg) < 0.20, f"Expected dark-red error banner bg, got {banner_bg}"
            assert _luminance(banner_fg) > 0.65, f"Expected light error banner text, got {banner_fg}"
        else:
            assert _luminance(banner_bg) > 0.85, f"Expected light-red error banner bg, got {banner_bg}"
            assert _luminance(banner_fg) < 0.35, f"Expected dark-red error banner text, got {banner_fg}"

        # Dismiss error banner cleanly
        page.locator(".error-banner-dismiss").click()
        banner.wait_for(state="hidden", timeout=5_000)
        assert errors == []
    finally:
        context.close()


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_theme_visual_styling_populated_results(
    browser,
    live_server_url: str,
    theme_e2e_assets: dict[str, Any],
    theme_name: str,
):
    """Verify capture hint, rendered Meeting Notes, Screenshots gallery, and Chat chips in each theme."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        color_scheme=theme_name,
    )
    context.add_init_script(f"localStorage.setItem('meeting-notes-theme', '{theme_name}');")
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_server_url, wait_until="domcontentloaded")
        _wait_for_theme(page, theme_name)

        # Upload dummy Recording -> verify capture-hint-box appears with theme-matched styling
        file_inputs = page.locator("input[type='file']")
        file_inputs.first.set_input_files(theme_e2e_assets["recording"])
        hint_box = page.locator(".capture-hint-box")
        hint_box.wait_for(state="visible", timeout=10_000)
        hint_bg = _effective_bg(page, ".capture-hint-box")
        hint_fg = _computed_style(page, ".capture-hint-box", "color")
        if theme_name == "dark":
            assert _luminance(hint_bg) < 0.30, f"Expected dark capture hint bg, got {hint_bg}"
            assert _luminance(hint_fg) > 0.65, f"Expected light capture hint text, got {hint_fg}"
        else:
            assert _luminance(hint_bg) > 0.75, f"Expected light capture hint bg, got {hint_bg}"
            assert _luminance(hint_fg) < 0.35, f"Expected dark capture hint text, got {hint_fg}"

        # Generate Meeting Notes
        page.get_by_role("button", name="Generate Meeting Notes").click()
        rendered_article = page.locator(".prose.rendered-notes-html article")
        rendered_article.wait_for(state="visible", timeout=15_000)
        assert "Executive Summary" in rendered_article.inner_text()

        # 1. Rendered Meeting Notes background & text contrast
        notes_bg = _effective_bg(page, ".prose.rendered-notes-html")
        notes_fg = _computed_style(page, ".prose.rendered-notes-html article", "color")
        if theme_name == "dark":
            assert _luminance(notes_bg) < 0.25, f"Expected dark rendered notes bg, got {notes_bg}"
            assert _luminance(notes_fg) > 0.65, f"Expected light rendered notes text, got {notes_fg}"
        else:
            assert _luminance(notes_bg) > 0.75, f"Expected light rendered notes bg, got {notes_bg}"
            assert _luminance(notes_fg) < 0.35, f"Expected dark rendered notes text, got {notes_fg}"

        # 2. Screenshots tab: verify dark mode does NOT have a bright white gallery background
        screenshots_tab = page.get_by_role("tab", name="Screenshots")
        screenshots_tab.wait_for(state="visible", timeout=5_000)
        screenshots_tab.click()
        gallery = page.locator(".screenshots-gallery").first
        gallery.wait_for(state="visible", timeout=5_000)
        gallery_bg = _effective_bg(page, ".screenshots-gallery")
        if theme_name == "dark":
            assert _luminance(gallery_bg) < 0.30, (
                f"Screenshots gallery should not render a bright white box in dark mode, got {gallery_bg}"
            )

        # 3. Meeting Chat tab: verify starter question chips are visible and contrasted
        page.get_by_role("tab", name="Meeting Chat").click()
        starter_chip = page.locator(".chat-starter-btn").first
        starter_chip.wait_for(state="visible", timeout=5_000)
        chip_bg = _effective_bg(page, ".chat-starter-btn")
        chip_fg = _computed_style(page, ".chat-starter-btn", "color")
        if theme_name == "dark":
            assert _luminance(chip_bg) < 0.35, f"Expected dark starter chip bg, got {chip_bg}"
            assert _luminance(chip_fg) > 0.65, f"Expected light starter chip text, got {chip_fg}"
        else:
            assert _luminance(chip_bg) > 0.70, f"Expected light starter chip bg, got {chip_bg}"
            assert _luminance(chip_fg) < 0.35, f"Expected dark starter chip text, got {chip_fg}"

        assert errors == []
    finally:
        context.close()
