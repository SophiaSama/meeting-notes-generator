"""Playwright browser E2E tests for the Gradio Export to Obsidian UI panel."""

from __future__ import annotations

import base64
from datetime import datetime
import io
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Callable

from PIL import Image
import pytest

from meeting_notes.bootstrap import ProxyDefaults
from meeting_notes.notes import NotesService
from meeting_notes.pipeline import PipelineService
from meeting_notes.ui import CSS, THEME, THEME_HEAD_HTML, MeetingNotesApp
from meeting_notes.video import VideoService
from tests.conftest import FakeLLMService, _write_fake_transcript

pytestmark = [pytest.mark.playwright, pytest.mark.e2e]

playwright_sync = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright is required to run browser E2E tests",
)
sync_playwright = playwright_sync.sync_playwright
expect = playwright_sync.expect

_CANNED_NOTES = """# Q4 Platform Launch Sync

## Executive Summary
The team agreed to ship the Obsidian vault export workflow.
@@SCREENSHOT_1@@

## Decisions Made
- Preserve unrelated files on confirmed overwrite.

## Action Items
- **Alice**: finalize the export documentation.
@@SCREENSHOT_2@@
"""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_browser(p: Any) -> Any:
    try:
        return p.chromium.launch(headless=True)
    except Exception:
        return p.chromium.launch(headless=True, channel="chrome")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _poll_until(
    page: Any,
    predicate: Callable[[], bool],
    *,
    timeout_ms: int = 10_000,
    interval_ms: int = 100,
    failure_msg: str = "Timed out waiting for condition",
) -> None:
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise AssertionError(failure_msg)
        page.wait_for_timeout(interval_ms)


class _ObsidianUiFakeLLMService(FakeLLMService):
    """Stage-aware FakeLLMService supporting notes, title, FAQ, and Meeting Chat Q&A."""

    def __init__(self):
        super().__init__(reply=_CANNED_NOTES)
        self.qa_reply = "Alice owns finalizing the export documentation."
        self.faq_reply = "## FAQ\n\n### Who owns export docs?\nAlice owns the export documentation."

    def call_llm(self, messages, **kwargs):  # type: ignore[override]
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        stage = str(kwargs.get("stage") or "").lower()
        system_prompt = str(kwargs.get("system_prompt") or "").lower()
        if "faq" in stage or "faq" in system_prompt or "distill" in system_prompt:
            return self.faq_reply
        if stage == "meeting_qa" or "grounded meeting q&a" in system_prompt:
            return self.qa_reply
        return self.reply


class _ObsidianUiVideoService(VideoService):
    """Deterministic video service returning 2 PIL screenshots without ffmpeg/cv2."""

    def __init__(self):
        pass

    def get_video_info(self, video_path: str) -> dict[str, Any]:
        return {
            "filename": Path(video_path).name,
            "duration": 120.0,
            "duration_str": "2:00",
            "size_mb": "0.10",
            "resolution": "1280x720",
            "fps": 30.0,
        }

    def sample_frame_bytes(self, *args: Any, **kwargs: Any) -> list[int]:
        return []

    def extract_frames(self, video_path: str, interval_seconds: int, max_frames: int):
        img1 = Image.new("RGB", (320, 180), color=(40, 80, 140))
        img2 = Image.new("RGB", (320, 180), color=(90, 50, 120))
        return [
            {
                "index": 1,
                "time_seconds": 2,
                "time_str": "0:02",
                "pil_image": img1,
                "base64_jpeg": _pil_to_b64(img1),
                "ocr_text": "",
                "ocr_keywords": [],
            },
            {
                "index": 2,
                "time_seconds": 4,
                "time_str": "0:04",
                "pil_image": img2,
                "base64_jpeg": _pil_to_b64(img2),
                "ocr_text": "",
                "ocr_keywords": [],
            },
        ]

    def transcribe_video_chunked(self, video_path: str, **kwargs: Any) -> dict[str, Any]:
        return _write_fake_transcript(video_path)


def _fill_settled(
    page: Any,
    locator: Any,
    value: str,
    *,
    settle_ms: int = 0,
    timeout_s: float = 5.0,
) -> None:
    locator.wait_for(state="visible", timeout=int(timeout_s * 1000))
    if settle_ms > 0:
        page.wait_for_timeout(settle_ms)
    deadline = time.monotonic() + timeout_s
    while True:
        locator.fill(value)
        page.wait_for_timeout(50)
        if locator.input_value() == value or time.monotonic() >= deadline:
            break
    expect(locator).to_have_value(value, timeout=int(timeout_s * 1000))


def _fill_chat_and_send(page: Any, question: str) -> None:
    page.get_by_role("tab", name="Meeting Chat").click()
    chat_box = page.get_by_placeholder("Ask a question about the meeting...")
    _fill_settled(page, chat_box, question, settle_ms=250, timeout_s=5.0)
    page.get_by_role("button", name="Send").click()


def _open_obsidian_panel(page: Any) -> None:
    vault_input = page.get_by_label(re.compile(r"Vault", re.I)).first
    if vault_input.is_visible():
        return
    page.get_by_role("button", name=re.compile(r"Export to Obsidian", re.I)).first.click()
    vault_input.wait_for(state="visible", timeout=10_000)
    page.wait_for_timeout(100)


def _wait_for_app_ready(page: Any) -> None:
    gen_btn = page.get_by_role("button", name="Generate Meeting Notes")
    gen_btn.wait_for(state="visible", timeout=10_000)
    expect(gen_btn).to_be_enabled(timeout=10_000)
    page.wait_for_function("() => window.__meetingNotesThemeSyncBound === true", timeout=10_000)
    if page.locator(".file-preview").count() == 0:
        expect(page.locator("input[type='file']")).to_have_count(2, timeout=10_000)


def _generate_notes_via_upload(page: Any, file_input_index: int, file_path: str) -> None:
    _wait_for_app_ready(page)
    accept_token = "video" if file_input_index == 0 else ".txt"
    target_input = page.locator(f"input[type='file'][accept*='{accept_token}']")
    expect(target_input).to_have_count(1, timeout=10_000)
    target_input.wait_for(state="attached", timeout=10_000)
    page.wait_for_timeout(150)

    uploaded_badge = page.get_by_text(Path(file_path).name, exact=False).first
    for attempt in range(3):
        if uploaded_badge.is_visible():
            break
        if attempt == 0 or target_input.count() > 0:
            target_input.set_input_files(file_path)
        try:
            uploaded_badge.wait_for(
                state="visible",
                timeout=4_000 if attempt < 2 else 10_000,
            )
            break
        except playwright_sync.TimeoutError:
            if attempt == 2:
                raise
            page.wait_for_timeout(200)

    if file_input_index == 0:
        page.locator(".capture-hint-box").or_(
            page.get_by_text("Duplicate video detected.", exact=False)
        ).first.wait_for(state="visible", timeout=10_000)

    gen_btn = page.get_by_role("button", name="Generate Meeting Notes")
    gen_btn.click()
    page.locator(".prose.rendered-notes-html article").wait_for(state="visible", timeout=20_000)
    expect(gen_btn).to_be_enabled(timeout=10_000)
    if file_input_index == 0:
        page.get_by_role("tab", name="Screenshots").wait_for(state="visible", timeout=10_000)
    page.wait_for_timeout(300)


@pytest.fixture(scope="module")
def live_obsidian_server(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("obsidian_ui_e2e")
    work_dir = root / "work"
    work_dir.mkdir()
    prev_cwd = Path.cwd()
    os.chdir(work_dir)

    dummy_recording = root / "q4_sync.mp4"
    dummy_recording.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    dummy_transcript = root / "q4_transcript.txt"
    dummy_transcript.write_text("Transcript content for Q4 Platform Launch Sync.", encoding="utf-8")

    defaults = ProxyDefaults(
        base_url="http://fake-provider.local/v1",
        api_key="sk-obsidian-ui-e2e-key",
        model="test-obsidian-ui-model",
        reasoning_effort="none",
    )
    mp = pytest.MonkeyPatch()
    mp.setattr(ProxyDefaults, "from_env", classmethod(lambda cls: defaults))

    llm = _ObsidianUiFakeLLMService()
    video = _ObsidianUiVideoService()
    notes = NotesService()
    pipeline = PipelineService(llm, video, notes)
    try:
        app = MeetingNotesApp(
            llm_service=llm,
            video_service=video,
            notes_service=notes,
            pipeline_service=pipeline,
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
        yield {
            "url": f"http://127.0.0.1:{port}",
            "recording": str(dummy_recording),
            "transcript": str(dummy_transcript),
        }
    finally:
        app.demo.close()
        os.chdir(prev_cwd)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = _launch_browser(p)
        try:
            yield b
        finally:
            b.close()


def test_obsidian_panel_guarded_initially_and_unlocked_after_generation(
    browser, live_obsidian_server: dict[str, Any]
):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_obsidian_server["url"], wait_until="domcontentloaded")
        page.locator(".empty-state-card").wait_for(state="visible", timeout=10_000)
        _wait_for_app_ready(page)

        expect(page.get_by_role("button", name="Prepare Obsidian Export")).not_to_be_visible()
        expect(page.get_by_role("button", name="Confirm & Export to Obsidian")).not_to_be_visible()

        _generate_notes_via_upload(page, 1, live_obsidian_server["transcript"])
        _open_obsidian_panel(page)

        expect(page.get_by_label(re.compile(r"Vault", re.I)).first).to_be_visible()
        expect(page.get_by_role("button", name="Browse...")).to_be_visible()
        expect(page.get_by_label(re.compile(r"Prefix", re.I)).first).to_be_visible()
        prepare_btn = page.get_by_role("button", name="Prepare Obsidian Export")
        expect(prepare_btn).to_be_visible()
        expect(prepare_btn).to_be_enabled()
        expect(page.get_by_role("button", name="Confirm & Export to Obsidian")).not_to_be_visible()
        expect(page.get_by_role("button", name="Cancel")).not_to_be_visible()
        expect(page.get_by_label(re.compile(r"FAQ", re.I)).first).not_to_be_visible()
        assert errors == []
    finally:
        context.close()


def test_browse_button_populates_vault_and_handles_headless_fallback_in_browser(
    browser, live_obsidian_server: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    vault = tmp_path / "PickedVault"
    vault.mkdir()

    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_obsidian_server["url"], wait_until="domcontentloaded")
        _generate_notes_via_upload(page, 1, live_obsidian_server["transcript"])
        _open_obsidian_panel(page)

        vault_input = page.get_by_label(re.compile(r"Vault", re.I)).first
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=f"{vault}/\n", stderr=""),
        )
        page.get_by_role("button", name="Browse...").click()
        expect(vault_input).to_have_value(str(vault), timeout=10_000)

        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(["osascript"], 60)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        page.get_by_role("button", name="Browse...").click()
        page.locator("#obsidian-status-md").get_by_text(re.compile(r"manually", re.I)).first.wait_for(
            state="visible", timeout=10_000
        )
        expect(vault_input).to_have_value(str(vault))
        assert errors == []
    finally:
        context.close()


def test_recording_chat_prepare_edit_faq_and_confirm_export(
    browser, live_obsidian_server: dict[str, Any], tmp_path: Path
):
    vault = tmp_path / "RecordingVault"
    vault.mkdir()

    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_obsidian_server["url"], wait_until="domcontentloaded")
        _generate_notes_via_upload(page, 0, live_obsidian_server["recording"])

        _fill_chat_and_send(page, "Who owns export docs?")
        page.get_by_text("Alice owns finalizing the export documentation.", exact=False).first.wait_for(
            state="visible", timeout=10_000
        )

        _open_obsidian_panel(page)
        _fill_settled(page, page.get_by_label(re.compile(r"Vault", re.I)).first, str(vault))
        _fill_settled(page, page.get_by_label(re.compile(r"Prefix", re.I)).first, "q4 slide")
        page.get_by_role("button", name="Prepare Obsidian Export").click()

        faq_box = page.get_by_label(re.compile(r"FAQ", re.I)).first
        faq_box.wait_for(state="visible", timeout=10_000)
        expect(faq_box).to_have_value(re.compile(r"Who owns export docs", re.I), timeout=10_000)

        edited_faq = "## Edited Browser FAQ\n\n### Owner\nAlice."
        _fill_settled(page, faq_box, edited_faq)

        confirm_btn = page.get_by_role("button", name="Confirm & Export to Obsidian")
        confirm_btn.wait_for(state="visible", timeout=10_000)
        confirm_btn.click()

        def _exported() -> bool:
            dirs = [p for p in vault.iterdir() if p.is_dir()]
            if len(dirs) != 1:
                return False
            d = dirs[0]
            return (
                (d / f"{d.name}.md").is_file()
                and (d / f"{d.name} - FAQ.md").is_file()
                and (d / "attachment" / "q4 slide_01.jpg").is_file()
                and (d / "attachment" / "q4 slide_02.jpg").is_file()
            )

        _poll_until(page, _exported, failure_msg="Obsidian Recording export did not complete")
        tdir = [p for p in vault.iterdir() if p.is_dir()][0]
        notes_md = (tdir / f"{tdir.name}.md").read_text(encoding="utf-8")
        assert "attachment/q4%20slide_01.jpg" in notes_md
        assert "attachment/q4%20slide_02.jpg" in notes_md
        assert (tdir / f"{tdir.name} - FAQ.md").read_text(encoding="utf-8").strip() == edited_faq

        page.get_by_role("tab", name="Rendered Meeting Notes").click()
        article = page.locator(".prose.rendered-notes-html article")
        article.wait_for(state="visible", timeout=5_000)
        assert "Executive Summary" in article.inner_text()
        assert errors == []
    finally:
        context.close()


def test_transcript_only_empty_chat_omits_faq_and_attachment_dir(
    browser, live_obsidian_server: dict[str, Any], tmp_path: Path
):
    vault = tmp_path / "TranscriptVault"
    vault.mkdir()

    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_obsidian_server["url"], wait_until="domcontentloaded")
        _generate_notes_via_upload(page, 1, live_obsidian_server["transcript"])
        _open_obsidian_panel(page)

        _fill_settled(page, page.get_by_label(re.compile(r"Vault", re.I)).first, str(vault))
        page.get_by_role("button", name="Prepare Obsidian Export").click()

        confirm_btn = page.get_by_role("button", name="Confirm & Export to Obsidian")
        confirm_btn.wait_for(state="visible", timeout=10_000)
        expect(page.get_by_label(re.compile(r"FAQ", re.I)).first).not_to_be_visible()
        confirm_btn.click()

        def _exported() -> bool:
            dirs = [p for p in vault.iterdir() if p.is_dir()]
            return len(dirs) == 1 and (dirs[0] / f"{dirs[0].name}.md").is_file()

        _poll_until(page, _exported, failure_msg="Transcript-only export did not complete")
        tdir = [p for p in vault.iterdir() if p.is_dir()][0]
        assert not (tdir / "attachment").exists()
        assert not (tdir / f"{tdir.name} - FAQ.md").exists()
        assert errors == []
    finally:
        context.close()


def test_collision_warning_cancel_then_confirm_overwrite_preserves_user_file(
    browser, live_obsidian_server: dict[str, Any], tmp_path: Path
):
    vault = tmp_path / "CollisionVault"
    vault.mkdir()
    folder_name = f"{datetime.now().strftime('%Y-%m-%d')} Q4 Platform Launch Sync"
    tdir = vault / folder_name
    tdir.mkdir()
    existing_note = tdir / f"{folder_name}.md"
    existing_note.write_text("OLD_NOTES", encoding="utf-8")
    user_file = tdir / "personal_notes.md"
    user_file.write_text("Keep me untouched", encoding="utf-8")

    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_obsidian_server["url"], wait_until="domcontentloaded")
        _generate_notes_via_upload(page, 1, live_obsidian_server["transcript"])
        _open_obsidian_panel(page)

        _fill_settled(page, page.get_by_label(re.compile(r"Vault", re.I)).first, str(vault))
        page.get_by_role("button", name="Prepare Obsidian Export").click()

        warning_box = page.locator("#obsidian-warning-md")
        warning_box.wait_for(state="visible", timeout=10_000)
        expect(warning_box).to_contain_text(
            re.compile(r"already exists|overwrite", re.I),
            timeout=10_000,
        )
        confirm_btn = page.get_by_role("button", name="Confirm & Export to Obsidian")
        confirm_btn.wait_for(state="visible", timeout=10_000)

        page.get_by_role("button", name="Cancel").click()
        confirm_btn.wait_for(state="hidden", timeout=10_000)
        assert existing_note.read_text(encoding="utf-8") == "OLD_NOTES"
        assert user_file.read_text(encoding="utf-8") == "Keep me untouched"

        page.get_by_role("button", name="Prepare Obsidian Export").click()
        warning_box.wait_for(state="visible", timeout=10_000)
        expect(warning_box).to_contain_text(
            re.compile(r"already exists|overwrite", re.I),
            timeout=10_000,
        )
        confirm_btn.wait_for(state="visible", timeout=10_000)
        confirm_btn.click()

        _poll_until(
            page,
            lambda: "The team agreed to ship" in existing_note.read_text(encoding="utf-8"),
            failure_msg="Overwrite export did not update notes file",
        )
        confirm_btn.wait_for(state="hidden", timeout=10_000)
        assert user_file.read_text(encoding="utf-8") == "Keep me untouched"
        assert errors == []
    finally:
        context.close()


def test_reset_and_new_generation_clear_pending_export(
    browser, live_obsidian_server: dict[str, Any], tmp_path: Path
):
    vault = tmp_path / "ResetVault"
    vault.mkdir()

    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(live_obsidian_server["url"], wait_until="domcontentloaded")
        _generate_notes_via_upload(page, 0, live_obsidian_server["recording"])
        _open_obsidian_panel(page)

        _fill_settled(page, page.get_by_label(re.compile(r"Vault", re.I)).first, str(vault))
        page.get_by_role("button", name="Prepare Obsidian Export").click()
        confirm_btn = page.get_by_role("button", name="Confirm & Export to Obsidian")
        confirm_btn.wait_for(state="visible", timeout=10_000)

        _generate_notes_via_upload(page, 1, live_obsidian_server["transcript"])
        confirm_btn.wait_for(state="hidden", timeout=10_000)
        expect(page.get_by_role("button", name="Generate Meeting Notes")).to_be_enabled(timeout=10_000)
        page.get_by_role("tab", name="Screenshots").wait_for(state="visible", timeout=10_000)
        assert list(vault.iterdir()) == []

        _open_obsidian_panel(page)
        _fill_settled(page, page.get_by_label(re.compile(r"Vault", re.I)).first, str(vault))
        page.get_by_role("button", name="Prepare Obsidian Export").click()
        confirm_btn.wait_for(state="visible", timeout=10_000)

        page.get_by_role("button", name="Reset", exact=True).click()
        page.locator(".empty-state-card").wait_for(state="visible", timeout=10_000)
        confirm_btn.wait_for(state="hidden", timeout=10_000)
        assert list(vault.iterdir()) == []
        assert errors == []
    finally:
        context.close()
