import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable

from .bootstrap import ProxyDefaults, setup_runtime_environment

setup_runtime_environment()

import gradio as gr

from .llm import REASONING_EFFORT_CHOICES, LLMService
from .notes import NotesService
from .pipeline import PipelineService
from .themes import (
    APPLY_THEME_JS,
    DEFAULT_THEME,
    RESTORE_THEME_JS,
    build_ui_css,
    theme_choices,
    to_embedded_html,
)
from .video import VideoService
from .views import (
    STARTER_QUESTIONS,
    ResultsPaneState,
    begin_generation,
    complete_generation,
    fail_generation,
    format_elapsed_time,
    format_obsidian_export_success,
    format_obsidian_overwrite_warning,
    format_obsidian_prepare_status,
    format_stage_progress,
    map_progress_to_stage,
    render_empty_state,
    render_provider_chip,
    reset_results_pane,
)

ART_THEME_CSS = build_ui_css()

THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
)

CSS = ART_THEME_CSS + """
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
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.title-row > .title-heading {
    flex: 1 1 0% !important;
    width: auto !important;
    min-width: min(220px, 100%) !important;
}
.title-row > .title-heading h1 {
    margin: 0 !important;
}
.art-theme-dd { min-width: 240px; flex: 0 0 auto !important; }
.rendered-notes-html { color: var(--body-text-color); line-height: 1.75; font-size: 1.05rem; }
.rendered-notes-html h1 { font-size: 1.6rem; font-weight: 700; color: var(--block-title-text-color); }
.rendered-notes-html h2 { font-size: 1.2rem; font-weight: 700; margin: 1.5rem 0 .4rem; color: var(--block-title-text-color); }
.rendered-notes-html .meta { font-size: .9rem; color: var(--body-text-color-subdued); margin-bottom: 1rem;
    padding-bottom: .75rem; border-bottom: 1px solid var(--border-color-primary); }
.rendered-notes-html blockquote { border-left: 3px solid var(--color-accent); margin: 1rem 0; padding: .5rem 1rem;
    color: var(--body-text-color-subdued); background: var(--background-fill-secondary); border-radius: var(--block-radius); }
.rendered-notes-html .screenshot { margin: 1.25rem 0; border: 1px solid var(--border-color-primary);
    border-radius: var(--block-radius); overflow: hidden; }
.rendered-notes-html .screenshot img { width: 100%; max-height: 260px; object-fit: cover; display: block; }
.rendered-notes-html .screenshot figcaption { font-size: .85rem; padding: 6px 12px; color: var(--body-text-color-subdued); }
.theme-toggle-btn {
    width: fit-content !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
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
    .title-row { gap: 8px; }
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
    margin-bottom: 0 !important;
    flex: 0 0 auto !important;
    width: fit-content !important;
    min-width: 0 !important;
}
.title-row > .provider-chip-container .html-container {
    padding: 0 !important;
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
    background: rgba(16, 185, 129, 0.2) !important;
    color: #6ee7b7 !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
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
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    justify-content: flex-end;
}
.share-actions-row > button,
.share-actions-row > a {
    flex: 0 0 auto !important;
    width: fit-content !important;
    min-width: 0 !important;
}
.results-container {
    width: 100%;
}
/* Screenshots Gallery */
.screenshots-gallery,
.screenshots-gallery .gallery-container,
.screenshots-gallery .grid-wrap,
.screenshots-gallery .grid-container,
.screenshots-gallery .empty {
    background: var(--block-background-fill) !important;
    color: var(--body-text-color) !important;
}
.screenshots-gallery .thumbnail-item {
    background: var(--background-fill-secondary) !important;
    border-color: var(--border-color-primary) !important;
}
.screenshots-gallery .caption-label {
    background: var(--background-fill-secondary) !important;
    color: var(--body-text-color) !important;
    border-color: var(--border-color-primary) !important;
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
.rendered-notes-html article h2 {
    text-transform: none !important;
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
/* Stage Progress Card */
.stage-progress-container {
    width: 100%;
    margin-bottom: 8px;
}
.stage-progress {
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary, #cbd5e1);
    border-radius: var(--block-radius, 8px);
    padding: 18px 20px;
    width: 100%;
    box-sizing: border-box;
    margin-bottom: 8px;
}
:root.dark .stage-progress, .dark .stage-progress {
    border-color: var(--border-color-primary, #475569);
}
.stage-progress.stage-done {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.35);
}
:root.dark .stage-progress.stage-done, .dark .stage-progress.stage-done {
    background: rgba(16, 185, 129, 0.14) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
}
.stage-progress-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-color-primary, #cbd5e1);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
}
:root.dark .stage-progress-header, .dark .stage-progress-header {
    border-bottom-color: var(--border-color-primary, #475569);
}
.stage-name {
    color: var(--body-text-color);
}
.stage-progress.stage-done .stage-name {
    color: #065f46;
}
:root.dark .stage-progress.stage-done .stage-name,
.dark .stage-progress.stage-done .stage-name {
    color: #6ee7b7 !important;
}
.stage-spinner {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
    border: 2px solid var(--color-accent);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.stage-check {
    font-weight: 700;
    font-size: 15px;
    line-height: 1;
    color: #10b981;
    flex-shrink: 0;
}
.stage-elapsed {
    margin-left: auto;
    color: var(--body-text-color-subdued);
    font-size: 12.5px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.stage-steps {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    list-style: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
.stage-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-color-primary, #cbd5e1);
    background: var(--background-fill-primary);
    font-size: 12.5px;
    line-height: 1.3;
    min-width: 0;
    list-style: none !important;
    margin: 0 !important;
}
:root.dark .stage-step, .dark .stage-step {
    border-color: var(--border-color-primary, #475569);
}
.stage-step-indicator {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 11px;
    font-weight: 600;
}
.stage-step-label {
    color: var(--body-text-color);
    font-weight: 500;
    overflow-wrap: anywhere;
}
.stage-step-completed {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.35);
}
:root.dark .stage-step-completed, .dark .stage-step-completed {
    background: rgba(16, 185, 129, 0.16) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
}
.stage-step-completed .stage-step-indicator {
    background: #10b981;
    color: #ffffff;
}
.stage-step-active {
    background: rgba(59, 130, 246, 0.08);
    border-color: var(--color-accent);
    box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.15);
}
:root.dark .stage-step-active, .dark .stage-step-active {
    background: rgba(59, 130, 246, 0.16) !important;
}
.stage-step-active .stage-step-indicator {
    border: 2px solid var(--color-accent);
    background: transparent;
}
.stage-step-active .stage-step-label {
    font-weight: 600;
}
.stage-step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--color-accent);
    animation: stage-pulse 1.4s ease-in-out infinite;
}
@keyframes stage-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(0.7); opacity: 0.5; }
}
.stage-step-pending {
    opacity: 0.72;
}
.stage-step-pending .stage-step-indicator {
    border: 1px solid var(--border-color-primary, #cbd5e1);
    color: var(--body-text-color-subdued);
}
:root.dark .stage-step-pending .stage-step-indicator,
.dark .stage-step-pending .stage-step-indicator {
    border-color: var(--border-color-primary, #475569);
}
.stage-step-pending .stage-step-label {
    color: var(--body-text-color-subdued);
}
.stage-step-skipped {
    border-style: dashed;
    opacity: 0.65;
}
.stage-step-skipped .stage-step-indicator {
    border: 1px dashed var(--border-color-primary, #cbd5e1);
    color: var(--body-text-color-subdued);
}
:root.dark .stage-step-skipped .stage-step-indicator,
.dark .stage-step-skipped .stage-step-indicator {
    border-color: var(--border-color-primary, #475569);
}
.stage-step-skipped .stage-step-label {
    color: var(--body-text-color-subdued);
}
.stage-step-badge {
    margin-left: auto;
    font-size: 10.5px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 9999px;
    background: var(--background-fill-secondary);
    color: var(--body-text-color-subdued);
    border: 1px solid var(--border-color-primary, #cbd5e1);
    white-space: nowrap;
}
:root.dark .stage-step-badge, .dark .stage-step-badge {
    border-color: var(--border-color-primary, #475569);
}
@media (max-width: 900px) {
    .stage-steps {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 520px) {
    .stage-steps {
        grid-template-columns: 1fr;
    }
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
    background: #fef2f2;
    border: 1px solid #f87171;
    color: #7f1d1d;
    box-sizing: border-box;
    width: 100%;
    margin-bottom: 12px;
    font-size: 13px;
    line-height: 1.45;
    position: relative;
    transition: opacity 0.2s ease-in-out;
}
.error-banner * {
    color: #7f1d1d;
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

THEME_HEAD_HTML = """<script>
(() => {
    try {
        const saved = localStorage.getItem('meeting-notes-theme');
        if (saved === 'dark' || saved === 'light') {
            document.documentElement.classList.toggle('dark', saved === 'dark');
            document.documentElement.style.colorScheme = saved;
        }
    } catch (e) {}
})();
</script>"""

INIT_THEME_JS = """() => {
    const KEY = 'meeting-notes-theme';
    const apply = (isDark) => {
        document.documentElement.classList.toggle('dark', isDark);
        if (document.body) document.body.classList.toggle('dark', isDark);
        document.documentElement.style.colorScheme = isDark ? 'dark' : 'light';
        const btn = document.querySelector('#theme-toggle-btn') || document.querySelector('.theme-toggle-btn');
        if (btn) {
            btn.textContent = isDark ? '🌙 Dark' : '☀️ Light';
            btn.setAttribute('aria-pressed', String(isDark));
            btn.setAttribute('aria-label', isDark ? 'Theme: Dark (switch to Light)' : 'Theme: Light (switch to Dark)');
        }
    };
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    const isDark = saved === 'dark'
        ? true
        : saved === 'light'
            ? false
            : (document.documentElement.classList.contains('dark')
                || (document.body && document.body.classList.contains('dark'))
                || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches));
    apply(isDark);
    if (!window.__meetingNotesThemeSyncBound) {
        window.__meetingNotesThemeSyncBound = true;
        window.addEventListener('storage', (ev) => {
            if (ev.key === KEY && (ev.newValue === 'dark' || ev.newValue === 'light')) {
                apply(ev.newValue === 'dark');
            }
        });
    }
}"""

TOGGLE_THEME_JS = """() => {
    const KEY = 'meeting-notes-theme';
    const isDark = document.documentElement.classList.contains('dark')
        || (document.body && document.body.classList.contains('dark'));
    const nextDark = !isDark;
    document.documentElement.classList.toggle('dark', nextDark);
    if (document.body) document.body.classList.toggle('dark', nextDark);
    document.documentElement.style.colorScheme = nextDark ? 'dark' : 'light';
    try { localStorage.setItem(KEY, nextDark ? 'dark' : 'light'); } catch (e) {}
    const btn = document.querySelector('#theme-toggle-btn') || document.querySelector('.theme-toggle-btn');
    if (btn) {
        btn.textContent = nextDark ? '🌙 Dark' : '☀️ Light';
        btn.setAttribute('aria-pressed', String(nextDark));
        btn.setAttribute('aria-label', nextDark ? 'Theme: Dark (switch to Light)' : 'Theme: Light (switch to Dark)');
    }
}"""

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


_TK_FOLDER_PICKER_SCRIPT = (
    "import sys, tkinter as tk\n"
    "from tkinter import filedialog\n"
    "root = tk.Tk()\n"
    "root.withdraw()\n"
    "try:\n"
    "    root.attributes('-topmost', True)\n"
    "except Exception:\n"
    "    pass\n"
    "init_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None\n"
    "chosen = filedialog.askdirectory(title='Select Obsidian Vault folder', initialdir=init_dir)\n"
    "root.destroy()\n"
    "if not chosen:\n"
    "    sys.exit(1)\n"
    "print(chosen)\n"
)

_MACOS_FOLDER_PICKER_SCRIPT = (
    "on run argv\n"
    "  if (count of argv) > 0 and item 1 of argv is not \"\" then\n"
    "    try\n"
    "      set defaultLoc to POSIX file (item 1 of argv) as alias\n"
    "      return POSIX path of (choose folder with prompt \"Select Obsidian Vault folder\" default location defaultLoc)\n"
    "    on error number -128\n"
    "      error number -128\n"
    "    on error\n"
    "      return POSIX path of (choose folder with prompt \"Select Obsidian Vault folder\")\n"
    "    end try\n"
    "  else\n"
    "    return POSIX path of (choose folder with prompt \"Select Obsidian Vault folder\")\n"
    "  end if\n"
    "end run\n"
)

_WINDOWS_FOLDER_PICKER_SCRIPT = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog; "
    "$dlg.Description = 'Select Obsidian Vault folder'; "
    "$dlg.ShowNewFolderButton = $true; "
    "if ($env:MEETING_NOTES_PICKER_INIT_DIR -and "
    "(Test-Path -LiteralPath $env:MEETING_NOTES_PICKER_INIT_DIR -PathType Container)) { "
    "$dlg.SelectedPath = $env:MEETING_NOTES_PICKER_INIT_DIR }; "
    "if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
    "Write-Output $dlg.SelectedPath } else { exit 1 }"
)


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
        self.last_html = format_stage_progress(
            stage,
            elapsed,
            transcript_only=self.is_transcript_only,
        )

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

    @staticmethod
    def _should_hide_capture_controls(video_path, transcript_file) -> bool:
        """Hide capture sliders and hint iff a Transcript is present and Recording is absent."""
        return bool(transcript_file) and not bool(video_path)

    def _on_video_input_change(self, video_path, transcript_file=None):
        """Handle upload updates in one callback to avoid extra duplicate work."""
        hide_controls = self._should_hide_capture_controls(video_path, transcript_file)
        duplicate_banner_update = self.pipeline_service.check_upload_for_duplicate(video_path)
        if isinstance(duplicate_banner_update, dict) and duplicate_banner_update.get("visible"):
            return (
                gr.update(visible=not hide_controls),
                gr.update(visible=not hide_controls),
                gr.update(value="", visible=False),
                duplicate_banner_update,
            )

        if hide_controls:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                duplicate_banner_update,
            )

        interval_update, max_frames_update, capture_hint_update = self._suggest_capture_settings(video_path)
        if isinstance(interval_update, dict):
            interval_update = {**interval_update, "visible": True}
        else:
            interval_update = gr.update(visible=True)
        if isinstance(max_frames_update, dict):
            max_frames_update = {**max_frames_update, "visible": True}
        else:
            max_frames_update = gr.update(visible=True)
        return interval_update, max_frames_update, capture_hint_update, duplicate_banner_update

    def _on_transcript_input_change(self, video_path, transcript_file):
        """Toggle capture control visibility on Transcript changes without re-sampling video."""
        if self._should_hide_capture_controls(video_path, transcript_file):
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            )
        hint_update = gr.update() if video_path else gr.update(value="", visible=False)
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            hint_update,
        )

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
        """Switch results pane into generating state and lock Generate Meeting Notes button."""
        state = begin_generation(is_transcript_only=bool(transcript_file and not video_file))
        empty_u, results_u, error_u, _starter_u, stage_u = cls._adapt_results_updates(state)
        return (
            empty_u,
            results_u,
            error_u,
            stage_u,
            gr.update(value="Generating...", interactive=False),
        )

    @staticmethod
    def _on_generate_finish(thumbnails):
        """Re-enable Generate button and show Screenshots tab only when screenshots exist."""
        has_screenshots = bool(thumbnails)
        return (
            gr.update(value="Generate Meeting Notes", interactive=True),
            gr.update(visible=has_screenshots),
            gr.update(selected="rendered-notes") if not has_screenshots else gr.update(),
        )

    @staticmethod
    def _sync_screenshots_tab(thumbnails):
        """Confirm Screenshots tab button visibility after TabItem mounts in Gradio 6."""
        return gr.update(visible=bool(thumbnails))

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
            self._wrap_html_preview(to_embedded_html(state.html_notes)),
            state.out_path,
            state.meeting_context,
            [],
            "",
            *self._adapt_results_updates(state),
        )

    @staticmethod
    def _wrap_html_preview(html_notes: str) -> str:
        """Ensure rendered Meeting Notes HTML exposes an <article> element inside gr.HTML."""
        if not html_notes or not html_notes.strip():
            return html_notes
        if "<article" in html_notes.lower():
            return html_notes
        if re.search(r"<body[^>]*>.*</body>", html_notes, flags=re.IGNORECASE | re.DOTALL):
            return re.sub(
                r"(<body[^>]*>)(.*)(</body>)",
                lambda m: f'{m.group(1)}<article class="meeting-notes-article">{m.group(2).strip()}</article>{m.group(3)}',
                html_notes,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            )
        return f'<article class="meeting-notes-article">{html_notes}</article>'

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
    def _on_reset_controls():
        """Restore Generate button, hide Screenshots tab, select Rendered Meeting Notes, and restore capture sliders."""
        return (
            gr.update(value="Generate Meeting Notes", interactive=True),
            gr.update(visible=False),
            gr.update(selected="rendered-notes"),
            gr.update(value=30, visible=True),
            gr.update(value=8, visible=True),
        )

    def _has_exportable_notes(self, meeting_context: Any) -> bool:
        """Return True iff meeting_context is valid and contains non-empty Meeting Notes."""
        if not isinstance(meeting_context, dict):
            return False
        if not self.pipeline_service._is_valid_meeting_context(meeting_context):
            return False
        return bool(
            str(meeting_context.get("raw_notes") or "").strip()
            or str(meeting_context.get("markdown_notes") or "").strip()
        )

    @staticmethod
    def _normalize_picked_directory(raw_stdout: str) -> str:
        """Trim picker stdout and remove trailing path separators except filesystem roots."""
        picked = (raw_stdout or "").strip()
        if not picked:
            return ""
        if len(picked) > 1 and picked.endswith(("/", "\\")):
            if not (len(picked) == 3 and picked[1] == ":" and picked[2] in ("/", "\\")):
                picked = picked.rstrip("/\\") or picked[0]
        return picked

    @staticmethod
    def _build_folder_picker_command(
        current_vault_path: str = "",
    ) -> tuple[list[str], dict[str, str] | None]:
        """Build a cross-platform short-lived subprocess command for selecting a folder."""
        initial_dir = ""
        candidate = (str(current_vault_path) if current_vault_path is not None else "").strip()
        if candidate:
            try:
                expanded = Path(candidate).expanduser()
                if expanded.is_dir():
                    initial_dir = str(expanded)
            except (OSError, ValueError, RuntimeError):
                initial_dir = ""

        if sys.platform == "darwin":
            cmd = ["osascript", "-e", _MACOS_FOLDER_PICKER_SCRIPT]
            if initial_dir:
                cmd.append(initial_dir)
            return cmd, None

        if sys.platform.startswith("win"):
            powershell_exe = (
                shutil.which("powershell")
                or shutil.which("pwsh")
                or "powershell"
            )
            cmd = [
                powershell_exe,
                "-NoProfile",
                "-NonInteractive",
                "-STA",
                "-Command",
                _WINDOWS_FOLDER_PICKER_SCRIPT,
            ]
            env = os.environ.copy()
            if initial_dir:
                env["MEETING_NOTES_PICKER_INIT_DIR"] = initial_dir
            return cmd, env

        zenity = shutil.which("zenity")
        if zenity:
            cmd = [
                zenity,
                "--file-selection",
                "--directory",
                "--title=Select Obsidian Vault folder",
            ]
            if initial_dir:
                cmd.append(f"--filename={initial_dir.rstrip('/')}/")
            return cmd, None

        kdialog = shutil.which("kdialog")
        if kdialog:
            start_dir = initial_dir
            if not start_dir:
                try:
                    start_dir = str(Path.home())
                except (OSError, RuntimeError):
                    start_dir = ""
            cmd = [kdialog, "--getexistingdirectory"]
            if start_dir:
                cmd.append(start_dir)
            cmd.extend(["--title", "Select Obsidian Vault folder"])
            return cmd, None

        cmd = [sys.executable, "-c", _TK_FOLDER_PICKER_SCRIPT]
        if initial_dir:
            cmd.append(initial_dir)
        return cmd, None

    def _on_obsidian_browse(self, current_vault_path: str = ""):
        """Open a native OS folder picker in a short-lived subprocess with manual fallback."""
        try:
            cmd, env = self._build_folder_picker_command(current_vault_path)
            run_kwargs: dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "timeout": 60,
            }
            if env is not None:
                run_kwargs["env"] = env
            result = subprocess.run(cmd, **run_kwargs)
        except subprocess.TimeoutExpired:
            return (
                gr.update(),
                gr.update(
                    value="Folder picker timed out. Enter or paste the Obsidian Vault Path manually.",
                    visible=True,
                ),
            )
        except Exception:
            return (
                gr.update(),
                gr.update(
                    value="Native folder picker unavailable in this environment. Enter or paste the Obsidian Vault Path manually.",
                    visible=True,
                ),
            )

        if getattr(result, "returncode", 1) == 0:
            picked = self._normalize_picked_directory(getattr(result, "stdout", "") or "")
            if picked:
                return (
                    gr.update(value=picked),
                    gr.update(value=f"Selected Obsidian Vault Path: `{picked}`", visible=True),
                )
            return (
                gr.update(),
                gr.update(
                    value="No folder selected. Enter or paste the Obsidian Vault Path manually.",
                    visible=True,
                ),
            )

        return (
            gr.update(),
            gr.update(
                value="Folder selection canceled. Existing Obsidian Vault Path kept.",
                visible=True,
            ),
        )

    def _on_obsidian_prepare(
        self,
        vault_path: str,
        attachment_prefix: str,
        meeting_context: Any,
        chat_history: Any,
        base_url: str,
        api_key: str,
        model: str,
        output_language: str | None = None,
        reasoning_effort: str | None = None,
    ):
        """Validate MeetingContext, run prepare_obsidian_export, and reveal preview/confirmation controls."""
        if not self._has_exportable_notes(meeting_context):
            return (
                None,
                gr.update(
                    value="Please generate Meeting Notes first before preparing an Obsidian export.",
                    visible=True,
                ),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        try:
            prepared = self.pipeline_service.prepare_obsidian_export(
                vault_path,
                meeting_context=meeting_context,
                chat_history=chat_history,
                base_url=base_url,
                api_key=api_key,
                model=model,
                output_language=output_language,
                reasoning_effort=reasoning_effort,
                attachment_prefix=attachment_prefix,
            )
        except Exception as exc:
            return (
                None,
                gr.update(value=f"⚠️ Could not prepare Obsidian export: {exc}", visible=True),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        faq_md = str(prepared.get("faq_markdown") or "").strip()
        has_faq = bool(faq_md)
        exists = bool(prepared.get("exists"))

        return (
            prepared,
            gr.update(value=format_obsidian_prepare_status(prepared), visible=True),
            (
                gr.update(
                    value=format_obsidian_overwrite_warning(str(prepared.get("target_dir") or "")),
                    visible=True,
                )
                if exists
                else gr.update(value="", visible=False)
            ),
            (
                gr.update(value=faq_md, visible=True, interactive=True)
                if has_faq
                else gr.update(value="", visible=False, interactive=True)
            ),
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
        )

    def _on_obsidian_confirm(
        self,
        pending_export: Any,
        attachment_prefix: str,
        faq_markdown: str,
    ):
        """Confirm and write the prepared Obsidian export using session-isolated gr.State."""
        if not isinstance(pending_export, dict) or not pending_export.get("vault_path"):
            return (
                None,
                gr.update(
                    value="No prepared Obsidian export found. Click Prepare Obsidian Export first.",
                    visible=True,
                ),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        overwrite = bool(pending_export.get("exists", False))
        cleaned_faq = faq_markdown if (faq_markdown is not None and str(faq_markdown).strip()) else ""

        try:
            result = self.pipeline_service.export_to_obsidian(
                pending_export,
                attachment_prefix=attachment_prefix,
                faq_markdown=cleaned_faq,
                overwrite=overwrite,
            )
        except Exception as exc:
            return (
                pending_export,
                gr.update(value=f"⚠️ Obsidian export failed: {exc}", visible=True),
                gr.update(),
                gr.update(),
                gr.update(visible=True, interactive=True),
                gr.update(visible=True, interactive=True),
            )

        if result.get("status") == "conflict" or not result.get("written"):
            updated_pending = {**pending_export, "exists": True}
            target_dir = str(result.get("target_dir") or pending_export.get("target_dir") or "")
            return (
                updated_pending,
                gr.update(
                    value="Target folder state changed before export. Review the warning below and confirm again or Cancel.",
                    visible=True,
                ),
                gr.update(
                    value=format_obsidian_overwrite_warning(target_dir),
                    visible=True,
                ),
                gr.update(),
                gr.update(visible=True, interactive=True),
                gr.update(visible=True, interactive=True),
            )

        return (
            None,
            gr.update(value=format_obsidian_export_success(result), visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    @staticmethod
    def _on_obsidian_cancel():
        """Discard pending Obsidian export state without calling PipelineService or writing disk."""
        return (
            None,
            gr.update(value="Obsidian export canceled. No files were written or modified.", visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _on_obsidian_after_generate(self, meeting_context: Any):
        """Sync Obsidian prepare guard and clear any stale pending export after generation."""
        can_export = self._has_exportable_notes(meeting_context)
        return (
            None,
            gr.update(interactive=can_export),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
        )

    @staticmethod
    def _on_obsidian_reset():
        """Reset Obsidian export controls and clear session-isolated pending export state."""
        return (
            None,
            gr.update(value=""),
            gr.update(value="screenshot"),
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(open=False),
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
            obsidian_export_state = gr.State(value=None)
            self.obsidian_export_state = obsidian_export_state

            is_configured = bool(self.defaults.api_key and self.defaults.api_key.strip())
            with gr.Row(elem_classes="title-row"):
                gr.Markdown("# Meeting Notes Helper", elem_classes=["title-heading"])
                provider_status_chip = gr.HTML(
                    value=render_provider_chip(
                        is_configured=is_configured,
                        model=self.defaults.model,
                    ),
                    elem_classes=["provider-chip-container"],
                )
                art_theme_dd = gr.Dropdown(
                    choices=theme_choices(),
                    value=DEFAULT_THEME,
                    label="Visual Theme",
                    show_label=False,
                    container=False,
                    scale=0,
                    min_width=240,
                    elem_classes=["art-theme-dd"],
                )
                theme_toggle_btn = gr.Button(
                    "☀️ Light",
                    size="sm",
                    variant="secondary",
                    scale=0,
                    elem_id="theme-toggle-btn",
                    elem_classes=["theme-toggle-btn"],
                )
            gr.Markdown(
                "Add a Recording or a Transcript -> generate structured Meeting Notes via any OpenAI-compatible model.",
                elem_classes="subtitle",
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
                    capture_hint = gr.HTML(visible="hidden")
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
                    dup_banner = gr.HTML(visible="hidden")
                    error_banner = gr.HTML(
                        value="",
                        visible="hidden",
                        elem_classes=["error-banner-container"],
                    )
                    stage_progress_html = gr.HTML(
                        value="",
                        visible="hidden",
                        elem_classes=["stage-progress-container"],
                    )
                    empty_state_container = gr.HTML(
                        value=render_empty_state(),
                        visible=True,
                        elem_classes=["results-empty-state-wrapper"],
                    )
                    with gr.Column(visible="hidden", elem_classes=["results-container"]) as results_container:
                        with gr.Tabs() as results_tabs:
                            with gr.TabItem("Rendered Meeting Notes", id="rendered-notes"):
                                with gr.Row(elem_classes=["share-actions-row"]):
                                    copy_md_btn = gr.Button(
                                        "Copy Markdown",
                                        elem_classes=["copy-md-btn", "copy-markdown-btn"],
                                        size="sm",
                                        scale=0,
                                    )
                                    download_btn = gr.DownloadButton(
                                        "Download HTML",
                                        value=None,
                                        elem_classes=["download-btn"],
                                        size="sm",
                                        scale=0,
                                    )
                                html_notes = gr.HTML(label="Rendered Meeting Notes", elem_classes=["rendered-notes-html"])
                            with gr.TabItem("Markdown", id="markdown-notes"):
                                notes_text = gr.Textbox(
                                    label="",
                                    lines=20,
                                    buttons=["copy"],
                                    elem_classes="note-box",
                                )
                            with gr.TabItem("Screenshots", id="screenshots-tab", visible=False) as screenshots_tab:
                                screenshots_gallery = gr.Gallery(
                                    label="Screenshots",
                                    columns=4,
                                    height="auto",
                                    show_label=False,
                                    elem_classes=["screenshots-gallery"],
                                )
                            with gr.TabItem("Meeting Chat", id="meeting-chat"):
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

                        with gr.Accordion(
                            "Export to Obsidian",
                            open=False,
                            elem_id="obsidian-export-accordion",
                            elem_classes=["obsidian-export-accordion"],
                        ) as obsidian_export_accordion:
                            with gr.Row():
                                obsidian_vault_input = gr.Textbox(
                                    label="Obsidian Vault Path",
                                    placeholder="/path/to/Obsidian/Vault",
                                    info="Local Obsidian vault folder where Meeting Notes will be saved.",
                                    interactive=True,
                                    scale=4,
                                    elem_id="obsidian-vault-input",
                                )
                                obsidian_browse_btn = gr.Button(
                                    "Browse...",
                                    variant="secondary",
                                    scale=0,
                                    min_width=110,
                                    elem_id="obsidian-browse-btn",
                                )
                            with gr.Row():
                                obsidian_prefix_input = gr.Textbox(
                                    label="Screenshot Prefix",
                                    value="screenshot",
                                    placeholder="screenshot",
                                    info="Filename prefix for exported Screenshot files inside attachment/.",
                                    interactive=True,
                                    scale=3,
                                    elem_id="obsidian-prefix-input",
                                )
                                obsidian_prepare_btn = gr.Button(
                                    "Prepare Obsidian Export",
                                    variant="primary",
                                    interactive=False,
                                    scale=0,
                                    elem_id="obsidian-prepare-btn",
                                )
                            obsidian_status_md = gr.Markdown(
                                value="",
                                visible="hidden",
                                elem_id="obsidian-status-md",
                            )
                            obsidian_warning_md = gr.Markdown(
                                value="",
                                visible="hidden",
                                elem_id="obsidian-warning-md",
                            )
                            obsidian_faq_preview = gr.Textbox(
                                label="Distilled Meeting Chat FAQ (Editable)",
                                value="",
                                lines=8,
                                max_lines=16,
                                visible="hidden",
                                interactive=True,
                                placeholder="Distilled Meeting Chat FAQ will appear here when Meeting Chat has conversation turns.",
                                elem_id="obsidian-faq-preview",
                            )
                            with gr.Row():
                                obsidian_confirm_btn = gr.Button(
                                    "Confirm & Export to Obsidian",
                                    variant="primary",
                                    visible="hidden",
                                    scale=0,
                                    elem_id="obsidian-confirm-btn",
                                )
                                obsidian_cancel_btn = gr.Button(
                                    "Cancel",
                                    variant="secondary",
                                    visible="hidden",
                                    scale=0,
                                    elem_id="obsidian-cancel-btn",
                                )

            self.video_input = video_input
            self.transcript_input = transcript_input
            self.interval_slider = interval_slider
            self.max_frames_slider = max_frames_slider
            self.results_tabs = results_tabs
            self.screenshots_tab = screenshots_tab
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
            self.obsidian_export_accordion = obsidian_export_accordion
            self.obsidian_vault_input = obsidian_vault_input
            self.obsidian_browse_btn = obsidian_browse_btn
            self.obsidian_prefix_input = obsidian_prefix_input
            self.obsidian_prepare_btn = obsidian_prepare_btn
            self.obsidian_status_md = obsidian_status_md
            self.obsidian_warning_md = obsidian_warning_md
            self.obsidian_faq_preview = obsidian_faq_preview
            self.obsidian_confirm_btn = obsidian_confirm_btn
            self.obsidian_cancel_btn = obsidian_cancel_btn
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
            demo.load(
                None,
                inputs=[],
                outputs=[],
                js=INIT_THEME_JS,
            )
            self.art_theme_dd = art_theme_dd
            art_theme_dd.change(
                None,
                inputs=[art_theme_dd],
                outputs=[],
                js=APPLY_THEME_JS,
            )
            demo.load(
                None,
                inputs=[],
                outputs=[art_theme_dd],
                js=RESTORE_THEME_JS,
            )
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
                    generate_btn,
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
            ).then(
                self._on_obsidian_after_generate,
                inputs=[meeting_context_state],
                outputs=[
                    obsidian_export_state,
                    obsidian_prepare_btn,
                    obsidian_confirm_btn,
                    obsidian_cancel_btn,
                    obsidian_faq_preview,
                    obsidian_warning_md,
                    obsidian_status_md,
                ],
                show_progress="hidden",
            ).then(
                self._sync_screenshots_tab,
                inputs=[screenshots_gallery],
                outputs=[screenshots_tab],
                show_progress="hidden",
            ).then(
                self._on_generate_finish,
                inputs=[screenshots_gallery],
                outputs=[generate_btn, screenshots_tab, results_tabs],
                show_progress="hidden",
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

            obsidian_browse_btn.click(
                self._on_obsidian_browse,
                inputs=[obsidian_vault_input],
                outputs=[obsidian_vault_input, obsidian_status_md],
                show_progress="hidden",
            )
            obsidian_prepare_btn.click(
                self._on_obsidian_prepare,
                inputs=[
                    obsidian_vault_input,
                    obsidian_prefix_input,
                    meeting_context_state,
                    chatbot,
                    base_url_input,
                    api_key_input,
                    model_input,
                    language_dd,
                    reasoning_effort_input,
                ],
                outputs=[
                    obsidian_export_state,
                    obsidian_status_md,
                    obsidian_warning_md,
                    obsidian_faq_preview,
                    obsidian_confirm_btn,
                    obsidian_cancel_btn,
                ],
            )
            obsidian_confirm_btn.click(
                self._on_obsidian_confirm,
                inputs=[
                    obsidian_export_state,
                    obsidian_prefix_input,
                    obsidian_faq_preview,
                ],
                outputs=[
                    obsidian_export_state,
                    obsidian_status_md,
                    obsidian_warning_md,
                    obsidian_faq_preview,
                    obsidian_confirm_btn,
                    obsidian_cancel_btn,
                ],
            )
            obsidian_cancel_btn.click(
                self._on_obsidian_cancel,
                inputs=[],
                outputs=[
                    obsidian_export_state,
                    obsidian_status_md,
                    obsidian_warning_md,
                    obsidian_faq_preview,
                    obsidian_confirm_btn,
                    obsidian_cancel_btn,
                ],
                show_progress="hidden",
            )

            video_input.change(
                self._on_video_input_change,
                inputs=[video_input, transcript_input],
                outputs=[interval_slider, max_frames_slider, capture_hint, dup_banner],
                show_progress="hidden",
            )
            transcript_input.change(
                self._on_transcript_input_change,
                inputs=[video_input, transcript_input],
                outputs=[interval_slider, max_frames_slider, capture_hint],
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
            ).then(
                self._on_reset_controls,
                inputs=[],
                outputs=[
                    generate_btn,
                    screenshots_tab,
                    results_tabs,
                    interval_slider,
                    max_frames_slider,
                ],
                show_progress="hidden",
            ).then(
                self._on_obsidian_reset,
                inputs=[],
                outputs=[
                    obsidian_export_state,
                    obsidian_vault_input,
                    obsidian_prefix_input,
                    obsidian_prepare_btn,
                    obsidian_confirm_btn,
                    obsidian_cancel_btn,
                    obsidian_faq_preview,
                    obsidian_warning_md,
                    obsidian_status_md,
                    obsidian_export_accordion,
                ],
                show_progress="hidden",
            )

        return demo

    def launch(self) -> None:
        server_name = os.getenv("MEETING_NOTES_HOST", "127.0.0.1")
        server_port = self.find_available_port(7860, 7870, host=server_name)
        self.demo.launch(
            css=CSS,
            theme=THEME,
            head=THEME_HEAD_HTML,
            server_name=server_name,
            server_port=server_port,
            share=False,
            show_error=True,
        )
