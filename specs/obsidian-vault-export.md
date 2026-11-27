# Obsidian Vault Export and Distilled FAQ Generation Specification

## Problem Statement

After generating `Meeting Notes` from a `Recording` or `Transcript` and asking follow-up questions in `Meeting Chat`, users of `Meeting Notes Helper` have no direct way to save those results into a local Obsidian vault. Currently, `Meeting Notes` remain inside internal artifact storage as raw Markdown or downloadable HTML, `Screenshot` files sit in a shared internal images directory, and insights from `Meeting Chat` are lost when the session resets. To archive a meeting in Obsidian, users must manually create folders, copy Markdown, move and rename `Screenshot` files, rewrite image links, and manually summarize their `Meeting Chat` history.

## Solution

Add an on-demand **Obsidian Vault Export** workflow to both the `Meeting Notes Helper` Gradio UI and the stateless MCP server. When triggered by the user, the system prepares an export targeting a user-chosen local Obsidian vault directory:

1. **Summary-Derived Naming with Date Prefix**: Derives a sanitized title from the first `# ` heading in `Meeting Notes` (falling back to the `Provider` for a concise title, then to the source `Recording` or `Transcript` filename stem) and formats the meeting folder and filenames as `YYYY-MM-DD <Title>/<YYYY-MM-DD <Title>>.md` and `YYYY-MM-DD <Title>/<YYYY-MM-DD <Title>> - FAQ.md`.
2. **Per-Meeting Attachments with Prefix Renaming**: Copies `Screenshot` files into `<vault>/<meeting_folder>/attachment/` using a user-specified naming prefix, and automatically rewrites all Markdown image references in the exported `Meeting Notes` to relative `attachment/<prefix>_NN.jpg` links. When no `Screenshot` files exist (such as `Transcript`-only runs), no `attachment/` directory is created.
3. **Provider-Distilled FAQ from `Meeting Chat`**: When `Meeting Chat` has non-empty conversation turns, uses the configured `Provider` to distill the Q&A history into a clean, separate FAQ Markdown file, and presents it in an editable preview before writing. When `Meeting Chat` is empty, the FAQ file is omitted.
4. **Overwrite-Only Collision Handling**: Checks whether the target meeting folder already exists, prompts the user for explicit confirmation before overwriting, and performs no writes if the user cancels. When confirmed, only the exported artifacts (`Meeting Notes` `.md`, FAQ `.md`, and `attachment/`) are replaced, leaving any unrelated user-created files inside that meeting folder untouched.

## User Stories

1. As a `Meeting Notes Helper` user, I want to export generated `Meeting Notes` directly into my local Obsidian vault on demand, so that I do not have to manually copy files and rewrite links after each meeting.
2. As a desktop user in the Gradio UI, I want to pick my Obsidian vault folder using a native OS folder dialog (and also be able to paste or edit the path in a textbox), so that I can select any local vault directory even if a native dialog is unavailable in a headless environment.
3. As a user, I want the exported meeting folder and Markdown filenames to be automatically named with a `YYYY-MM-DD <Title>` convention derived from the `Meeting Notes`, so that my vault stays organized chronologically by topic without manual renaming.
4. As a user, I want the title derivation to use the first `# ` heading in `Meeting Notes` when available and only call the `Provider` as a fallback, so that exports are fast and avoid unnecessary LLM calls when a clear title already exists.
5. As a user, I want `Screenshot` files from a `Recording` to be placed in an `attachment/` subfolder inside each meeting's own folder rather than a vault-wide shared folder, so that each meeting folder is self-contained and easy to move or delete.
6. As a user exporting a `Transcript`-only session with zero `Screenshot` files, I want the export to skip creating an empty `attachment/` folder, so that my vault stays clean.
7. As a user, I want to set a custom naming prefix for exported `Screenshot` files and have all Markdown image references in the exported `Meeting Notes` automatically updated to match, so that image filenames are meaningful and links never break in Obsidian.
8. As a user who asked follow-up questions in `Meeting Chat`, I want the `Provider` to distill my chat history into a separate `<YYYY-MM-DD <Title>> - FAQ.md` file rather than dumping raw noisy chat logs into the main `Meeting Notes`, so that key Q&A insights are preserved cleanly alongside the meeting notes.
9. As a user, I want to preview and edit the distilled FAQ Markdown before anything is written to my Obsidian vault, so that I can fix inaccuracies or remove unwanted questions before saving.
10. As a user who did not use `Meeting Chat` (or only saw inline error/placeholder messages), I want the FAQ Markdown file to be omitted from the export, so that empty or placeholder FAQ files are never created in my vault.
11. As a user re-exporting a meeting after refining `Meeting Chat` or changing the `Screenshot` prefix, I want the app to warn me when the target meeting folder already exists and require my explicit confirmation before overwriting, so that I never overwrite existing notes by accident.
12. As a user who cancels an overwrite prompt, I want zero files in my Obsidian vault to be created, modified, or deleted, so that canceling is completely safe.
13. As an Obsidian user who adds personal follow-up notes inside an exported meeting folder, I want confirmed overwrites to replace only the exported `Meeting Notes`, FAQ file, and `attachment/` subfolder, so that my unrelated personal files in that folder are preserved.
14. As a user re-exporting a meeting after clearing `Meeting Chat` or changing the `Screenshot` prefix, I want stale exported files (such as an old FAQ file or differently prefixed `Screenshot` files) to be removed during overwrite, so that outdated exports do not linger alongside new ones.
15. As a user or automation client calling the MCP server, I want to prepare and execute Obsidian exports through stateless MCP tools, so that external agents can export `Meeting Notes`, `Screenshot` files, and distilled `Meeting Chat` FAQs using the same rules as the UI.
16. As a user working with reasoning-capable `Provider` models, I want FAQ distillation and title fallback calls to respect the active reasoning effort and length-retry handling, so that export preparation does not fail with empty outputs on reasoning models.
17. As a user, I want exporting to Obsidian to copy `Screenshot` files without moving or deleting internal `Meeting Notes Helper` artifacts, so that the Gradio UI continues to render `Meeting Notes` and `Screenshot` previews normally after export.

## Implementation Decisions

### Deep Module & Seam Placement
* **Extend `PipelineService` as the single primary deep module**: Both the Gradio UI (`MeetingNotesApp`) and the MCP server already delegate orchestration to `PipelineService`. Adding the two-phase Obsidian export contract (`prepare_obsidian_export` and `export_to_obsidian`) to `PipelineService` keeps title derivation, `Meeting Chat` normalization/filtering, `Provider` FAQ distillation, path sanitization, collision detection, and selective overwrite in one shared, testable seam.
* **Preserve `NotesService` as the pure Markdown/link-rewriting helper**: Parameterize `NotesService.build_markdown_notes` (or add a shared placeholder-substitution parameter for target attachment directory and filename prefix) so internal artifact generation and Obsidian export share one `@@SCREENSHOT_N@@` replacement and URL-encoding path.

### Context & State Extensions
* **Extend `MeetingContext` produced by `PipelineService.run_pipeline`**:
  * Store `raw_notes` (the clean `Provider` Markdown body containing `@@SCREENSHOT_N@@` placeholders before internal `artifact_summary` or `.minutes.md` metadata headers are attached) and `screenshots_dir` (the directory holding saved `Screenshot` files, or `None` for `Transcript`-only runs).
  * This allows Obsidian export to render clean plain Markdown directly without scraping UI textboxes or stripping internal `source_video:` / `Screenshot Alignment Checks` sections.
* **Gradio UI Session State**:
  * Add session-isolated `gr.State` for the prepared Obsidian export payload (target folder path, collision flag, derived title, and distilled FAQ draft) so concurrent browser sessions never share export state.
  * Reset Obsidian export preview/confirmation state on global reset and when generating new `Meeting Notes`.

### Two-Phase Export Contract (`PipelineService`)
1. **`prepare_obsidian_export(...)`**:
   * **Inputs**: `vault_path`, `meeting_context` (or explicit `raw_notes` / `notes_markdown`, `source_stem`, and `screenshots_dir`), `chat_history`, `base_url`, `api_key`, `model`, `output_language=None`, `reasoning_effort=None`, `export_date=None`.
   * **Validation**: Verify `vault_path` is a non-empty, existing, writable directory before making any `Provider` calls.
   * **Title & Folder Derivation**:
     * Extract the first `# ` heading from the `Meeting Notes` body if present.
     * Otherwise, call the `Provider` (via `_call_llm_with_length_retry` and `llm_service.reasoning_effort(...)`) to generate a concise title from the `Meeting Notes`, falling back to `source_stem` if the `Provider` fails or returns empty text.
     * Sanitize the title for cross-platform filesystem safety (stripping `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`, leading/trailing dots, and `..` traversal sequences), clamp length, and format with `YYYY-MM-DD <Title>`.
   * **Collision Check**: Determine whether `<vault_path>/<YYYY-MM-DD Title>` already exists (`exists: bool`).
   * **FAQ Distillation**: Normalize `chat_history` with `_normalize_chat_history`, filter out inline error/placeholder assistant messages (such as `"Please generate meeting notes first..."`), and if non-empty, call the `Provider` to distill the conversation into clean Q&A Markdown (`faq_markdown`). If empty, set `faq_markdown = None`.
   * **Output**: Structured preparation result containing `vault_path`, `folder_name`, `target_dir`, `notes_filename`, `faq_filename`, `exists`, `faq_markdown`, `screenshot_count`, and `default_attachment_prefix`.
2. **`export_to_obsidian(...)`**:
   * **Inputs**: Preparation metadata (or explicit `vault_path`, `folder_name`, `raw_notes` / `notes_markdown`, `screenshots_dir`, `frames_metadata`), `attachment_prefix`, `faq_markdown` (possibly edited by the user, or `None`), and `overwrite: bool = False`.
   * **Collision Guard**: If `target_dir` already exists and `overwrite` is `False`, refuse to write and return a conflict status without modifying disk.
   * **Prefix Sanitization & Link Rewriting**: Sanitize `attachment_prefix` (with a safe default such as `screenshot` when blank), rewrite `@@SCREENSHOT_N@@` placeholders to `attachment/<sanitized_prefix>_NN.jpg` (URL-encoded when needed), and copy `Screenshot` files non-destructively into `<target_dir>/attachment/` only when at least one `Screenshot` exists.
   * **Selective Overwrite**: When `overwrite=True` on an existing folder, replace only the exported artifacts:
     * Overwrite `<YYYY-MM-DD Title>.md` with the plain Markdown `Meeting Notes`.
     * Overwrite `<YYYY-MM-DD Title> - FAQ.md` if `faq_markdown` is non-empty, or remove an existing `<YYYY-MM-DD Title> - FAQ.md` if `faq_markdown` is now empty.
     * Replace `<target_dir>/attachment/` cleanly so stale `Screenshot` files from an earlier prefix do not linger (and remove `attachment/` if there are zero `Screenshot` files).
     * Leave any other user-created files or subdirectories inside `target_dir` untouched.

### Adapter Interfaces
* **Gradio UI (`MeetingNotesApp`)**:
  * Add an **Export to Obsidian** panel/section (disabled or guarded until `Meeting Notes` are generated) containing:
    * Vault Path textbox paired with a **Browse...** button that invokes a native OS folder dialog in a short-lived subprocess (avoiding main-thread AppKit/Tk crashes in Gradio worker threads, with graceful fallback to manual path entry).
    * `Screenshot` naming prefix textbox.
    * **Prepare Obsidian Export** trigger button -> runs `prepare_obsidian_export`, displays the target folder path, shows an explicit overwrite warning if `exists` is `True`, and reveals an editable **FAQ Preview** textbox when `Meeting Chat` produced a distilled FAQ.
    * **Confirm & Export to Obsidian** and **Cancel** buttons -> writes only on explicit confirmation (`overwrite=True` when `exists` is `True`); clicking **Cancel** clears the pending export state without touching disk.
  * All UI labels and messages must follow `GLOSSARY.md` terminology.
* **MCP Server**:
  * Expose stateless tool(s) wrapping `prepare_obsidian_export` and `export_to_obsidian` (e.g., `prepare_obsidian_export` and `export_to_obsidian` accepting `minutes_file` or `notes_markdown` + `screenshots_dir`, `vault_path`, optional `chat_history` / `faq_markdown`, `attachment_prefix`, and `overwrite: bool = False`).
  * Add a body-extraction helper for saved `.minutes.md` files so MCP exports strip internal `# Meeting Minutes` headers and `## Screenshot Alignment Checks` before writing plain Markdown to Obsidian.

## Testing Decisions

* **What Makes a Good Test**:
  * Tests exercise external behavior at the `PipelineService` seam (and thin adapter contracts for UI handlers and MCP tools), asserting on returned preparation payloads, written Obsidian vault directory trees, Markdown link targets, distilled FAQ contents, and untouched user files—never testing private prompt builders or internal helper functions in isolation.
* **Seams & Modules to Test**:
  * **Primary Seam (`PipelineService`)**: Use the existing `FakeLLMService` (at the `Provider` boundary) and `tmp_path` (for source `Screenshot` files and destination Obsidian vaults) to cover:
    1. Heading-based title extraction vs. `Provider` fallback title derivation and cross-platform filename sanitization.
    2. `Meeting Chat` distillation into `<YYYY-MM-DD Title> - FAQ.md`, omission when `Meeting Chat` is empty or contains only inline error placeholders, and preservation of user-edited `faq_markdown`.
    3. `Screenshot` copying into `attachment/` with custom prefix renaming and updated relative Markdown links, plus skipping `attachment/` when `Screenshot` count is zero.
    4. Collision detection (`exists=True`), refusal to write when `overwrite=False`, and selective overwrite when `overwrite=True` (removing stale prefixed images/FAQ while preserving unrelated user files in the meeting folder).
    5. Non-destructive export (internal `artifacts/meeting_minutes/` files remain intact) and invalid/unwritable vault path error handling.
  * **Thin Adapter Seams**:
    * UI handler tests (following `test_ui_handlers.py` patterns): verify Gradio state updates across Prepare -> Preview/Edit -> Confirm/Cancel, reset clearing, and mocked subprocess OS folder dialog fallback.
    * MCP server tests (following `test_mcp_server.py` patterns): verify stateless MCP export tools extract clean `Meeting Notes` from `.minutes.md` files, pass optional `chat_history`/`faq_markdown`, and enforce `overwrite=False` collision protection.
* **Prior Art**:
  * `tests/conftest.py` (`FakeLLMService`, `FakeVideoService`), `tests/test_e2e_pipeline.py`, `tests/test_meeting_qa_chatbot.py`, `tests/test_ui_handlers.py`, and `tests/test_mcp_server.py`.

## Out of Scope

* Merging, appending, or auto-versioning existing Obsidian notes on folder collision.
* Shared vault-wide attachment folders across multiple meetings or trainings.
* Automatic background syncing or exporting on every `Meeting Chat` turn.
* Obsidian-specific `![[...]]` wikilinks, custom YAML frontmatter templates, or Obsidian Local REST API integration.
* Remote or containerized browser-to-host filesystem bridging (assumes local execution on the same machine as the vault).

## Further Notes

* Brainstorm RFC: [obsidian-vault-export.md](file:///Users/ruiping/.gemini/antigravity/worktrees/meeting-notes-generator/export_notes_obsidian/docs/designs/obsidian-vault-export.md)
* Product terminology reference: [GLOSSARY.md](file:///Users/ruiping/.gemini/antigravity/worktrees/meeting-notes-generator/export_notes_obsidian/GLOSSARY.md)
