# Obsidian Vault Export and Distilled FAQ Generation

* **Author:** rpwang
* **Date:** 2026-11-22

## Overview

Meeting Notes Helper users can generate structured `Meeting Notes` from a `Recording` or `Transcript` and ask follow-up questions in `Meeting Chat`, but the outputs currently stay inside `artifacts/meeting_minutes/` as raw Markdown or downloadable HTML. This feature adds an on-demand export to a user-selected local Obsidian vault from both the Gradio UI and `mcp_server.py`.

Each export creates a meeting folder named from the meeting summary containing:
1. A plain Markdown `Meeting Notes` file named from the meeting summary.
2. A separate `Provider`-distilled FAQ Markdown file built from `Meeting Chat` (omitted when `Meeting Chat` is empty, and previewable/editable before writing).
3. `Screenshot` image files copied into a per-meeting `attachment/` subfolder using a user-chosen naming prefix, with all Markdown image links in the note automatically updated to match.

If the target meeting folder already exists, the app asks for confirmation before overwriting and makes no changes if the user cancels.

## Analysis

### Codebase Constraints
* **`meeting_notes/ui.py` vs. clean Markdown**: The Gradio `notes_text` box displays `raw_notes + artifact_summary`, while `MeetingContext["markdown_notes"]` holds `markdown_body` with `images/<stem>/screenshot_NN.jpg` links already substituted, and `artifacts/meeting_minutes/<stem>.minutes.md` prepends internal metadata (`source_video:`, `ocr_available:`, and `## Screenshot Alignment Checks`). Storing `raw_notes` and `screenshots_dir` (or saved frame paths) in `MeetingContext` lets Obsidian export produce clean plain Markdown and rewrite `@@SCREENSHOT_N@@` placeholders directly to `attachment/<prefix>_NN.jpg` without scraping UI summaries or internal headers.
* **Shared link-rewriting and LLM helpers**: `NotesService.build_markdown_notes` (`meeting_notes/notes.py`) already handles `@@SCREENSHOT_N@@` replacement and URL encoding. Parameterizing the attachment directory and filename prefix lets both internal artifact generation and Obsidian export share one code path. Similarly, `PipelineService._normalize_chat_history`, `PipelineService._call_llm_with_length_retry`, and `LLMService.reasoning_effort(...)` (`meeting_notes/pipeline.py`, `meeting_notes/llm.py`) can be reused for FAQ distillation and title derivation, filtering out inline error/placeholder messages (such as `"Please generate meeting notes first..."`) when checking whether `Meeting Chat` is empty.
* **Stateless `mcp_server.py`**: Because the MCP server does not hold Gradio `MeetingContext` or `Meeting Chat` session state across calls, the shared export service must accept explicit inputs (`minutes_file` or `notes_markdown` + `screenshots_dir`, plus optional `chat_history` or `faq_markdown`).

### Key Contradictions & Inventive Resolutions
* **Native OS folder dialog inside a browser UI**: Gradio runs in a browser and has no built-in directory picker, while calling `tkinter` or `osascript` directly inside a Gradio worker thread can hang or crash on macOS (where AppKit/Tk requires the main thread). **Resolution**: Launch the OS folder picker in a short-lived subprocess behind a "Browse..." button, paired with an editable Vault Path textbox so users can also paste or reuse a path if the subprocess fails in a headless environment.
* **Interactive FAQ preview + overwrite confirmation without losing atomicity**: Exporting involves two checkpoints—reviewing/editing the `Provider`-distilled FAQ when `Meeting Chat` is non-empty, and confirming overwrite when the target folder already exists—across both UI and MCP. **Resolution**: Split the export contract into two phases in one shared service:
  1. `prepare_obsidian_export`: Validates the vault path, derives the sanitized folder/file name, detects folder collision (`exists: bool`), and distills `Meeting Chat` into `faq_markdown` (or `None` if empty).
  2. `export_to_obsidian`: Accepts the (possibly edited) `faq_markdown`, `attachment_prefix`, and `overwrite: bool = False`. When `exists` is true and `overwrite` is false, it refuses to write; when confirmed (`overwrite=True`), it writes the `Meeting Notes` `.md`, optional FAQ `.md`, and `attachment/` `Screenshot` files cleanly without moving or deleting internal `artifacts/meeting_minutes/` files.

### Secondary Trade-offs & Edge Cases
* **Path and prefix sanitization**: Both `Provider`-derived meeting titles and user-entered `Screenshot` prefixes can contain characters invalid on macOS/Windows/Linux (`/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`) or `..` traversal sequences. Names must be sanitized and length-clamped with a deterministic fallback to the source `Recording`/`Transcript` stem.
* **Transcript-only runs (0 `Screenshot` files)**: When no `Recording` was processed, skip creating an empty `attachment/` directory.
* **Stale files vs. user files on overwrite**: If a user re-exports with a different `Screenshot` prefix or after clearing `Meeting Chat`, stale exported files could linger, while deleting the whole meeting folder would destroy unrelated notes the user added in Obsidian. Overwrite should replace only the exported artifacts (the `Meeting Notes` `.md`, the FAQ `.md`—deleting an old FAQ `.md` if `Meeting Chat` is now empty—and the `attachment/` subfolder) while leaving unrelated user files in the meeting folder untouched.
* **Terminology alignment**: All UI controls, status messages, and MCP tool descriptions must follow `GLOSSARY.md` (`Meeting Notes`, `Screenshot`, `Meeting Chat`, `Provider`, `Recording`, `Transcript`, `Notes Style`).

### Ideal Final Result
A single shared export service powers both the Gradio UI and `mcp_server.py`: users pick a local vault, preview/edit any distilled FAQ from `Meeting Chat`, adjust the `Screenshot` prefix, confirm overwrites if the folder exists, and get self-contained plain Markdown notes with per-meeting `attachment/` files—while internal `Meeting Notes Helper` artifacts remain untouched.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Local Obsidian vault selection in the Gradio UI via an OS folder dialog (with editable path textbox fallback).
  * Shared export service exposed to both `meeting_notes/ui.py` and `mcp_server.py`.
  * Summary-derived meeting folder name and `.md` filename.
  * Per-meeting `attachment/` subfolder for `Screenshot` images, with user-configurable naming prefix and automatic Markdown image link updates.
  * Separate `.md` file for a `Provider`-distilled FAQ from `Meeting Chat` (omitted when `Meeting Chat` is empty; previewable and editable before writing).
  * Overwrite-only collision handling with explicit user confirmation and cancel-safe behavior.
* **Out-of-Scope Non-Goals:**
  * Merging, appending, or auto-versioning existing Obsidian notes on collision.
  * Shared global attachment folders across multiple meetings or trainings.
  * Live background syncing or automatic export on every `Meeting Chat` turn.
  * Obsidian-specific `![[...]]` wikilinks, custom YAML frontmatter templates, or Obsidian Local REST API integration.
  * Remote/containerized browser-to-host filesystem bridging (assumes local execution on the same machine as the vault).

## Alternatives Weighed

* **Append raw `Meeting Chat` Q&A log to `Meeting Notes`**: Rejected by author because raw chat turns include exploratory or noisy exchanges; a separate `Provider`-distilled FAQ `.md` file keeps the main `Meeting Notes` clean and makes Q&A follow-ups independently readable.
* **Place all `Screenshot` files in a shared vault-wide attachments folder**: Rejected by author because per-meeting `attachment/` subfolders keep each meeting/training folder self-contained and easier to rename, move, or overwrite.
* **Auto-version or append on folder collision**: Rejected by author in favor of overwrite-only with an explicit confirmation prompt and no-op on cancel.
* **Web-only Gradio `FileExplorer` or text-only vault input**: Rejected by author in favor of an OS folder dialog for local vault selection (paired with a path textbox in the UI to handle subprocess/headless edge cases).

## Key Decisions & Staging for Next Steps

* **Resolved Decisions:**
  * Target environment is local execution for Gradio UI plus stateless tool exposure in `mcp_server.py`.
  * Export runs only on explicit user trigger, using a two-phase flow (prepare/preview/collision-check -> confirm/write).
  * Folder and filename convention includes a date prefix: `YYYY-MM-DD <Title>/<YYYY-MM-DD <Title>>.md` and `YYYY-MM-DD <Title>/<YYYY-MM-DD <Title>> - FAQ.md`.
  * Title derivation extracts the first `# ` heading from `Meeting Notes` when present and calls the `Provider` only as a fallback (with source filename stem as final fallback).
  * `Screenshot` files are copied (not moved) into `<vault>/<meeting_folder>/attachment/` with a user-supplied prefix and updated plain Markdown relative links.
  * `Meeting Chat` is distilled by the `Provider` into a separate FAQ `.md` file only when non-empty, and can be edited in preview before writing.
  * On confirmed overwrite, replace only the exported files (`Meeting Notes` `.md`, FAQ `.md`, and `attachment/`), preserving any unrelated user files inside the meeting folder.
