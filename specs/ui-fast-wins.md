## Problem Statement

Even after the initial UI polish, Meeting Notes Helper has several layout and state-clarity rough edges that hurt usability:

- The `Screenshots` gallery sits pinned above the results tabs, so when fewer than four `Screenshots` exist in dark mode the unused grid slots render as a bright white box, `Transcript`-only runs show an empty gallery box, and `Meeting Chat` loses 220px of vertical height (clipping the question input bar on 1080p screens).
- The header stacks title, subtitle, `Provider` status chip, and `Provider Settings` across four rows, while `Copy Markdown` and `Download HTML` stretch 50%/50% across the entire results pane.
- The theme toggle button does not indicate the currently active mode and resets on page reload.
- During generation, both the empty state and results container are hidden, leaving most of the right pane blank except for a small progress pill, while the `Generate Meeting Notes` button stays clickable and allows accidental double-submissions.
- When a user uploads a `Transcript` without a `Recording`, the `Screenshot interval` and `Max screenshots` sliders remain visible even though `Recording` processing is skipped.

## Solution

A fast-win UI and view-layer polish pass that reclaims vertical space, fixes dark-mode visual glitches, and makes input and generation states unambiguous—with zero changes to the pipeline, LLM, or video services. The `Provider` status chip moves inline into the title row, the theme toggle reflects and persists the active mode, `Screenshots` moves into its own tab (hidden on `Transcript`-only runs), share buttons become compact and right-aligned, generation locks the primary button and shows a 4-step progress card, and capture sliders hide automatically for `Transcript`-only input.

## User Stories

1. As a user, I want the `Provider` status chip inline in the header row next to the theme toggle, so that more of the `Meeting Notes` and `Meeting Chat` area fits above the fold.
2. As a user, I want the theme toggle button to show the active mode (`☀️ Light` or `🌙 Dark`) and remember my choice across reloads, so that I know which theme is active and do not have to re-toggle it.
3. As a user, I want `Screenshots` placed in its own tab (`Rendered Meeting Notes` | `Markdown` | `Screenshots` | `Meeting Chat`) rather than pinned above all tabs, so that `Rendered Meeting Notes` and `Meeting Chat` get the full vertical height of the results pane.
4. As a dark-mode user, I want the `Screenshots` gallery background to match the dark theme, so that unused grid slots do not appear as a stark white box.
5. As a user running `Meeting Notes Helper` from a `Transcript` only, I want the `Screenshots` tab hidden, so that I am not presented with an empty gallery.
6. As a user, I want `Copy Markdown` and `Download HTML` rendered as compact, right-aligned buttons, so that they do not dominate the top of `Rendered Meeting Notes`.
7. As a user, I want the `Generate Meeting Notes` button disabled and labelled `"Generating..."` while a run is in progress, so that I cannot accidentally trigger duplicate runs.
8. As a user waiting on generation, I want a 4-step visual progress card (`Extracting Screenshots` → `Transcribing` → `Summarizing` → `Done` with elapsed time) in the results pane, so that the right pane feels active and predictable.
9. As a user uploading a `Transcript` with no `Recording`, I want `Screenshot interval`, `Max screenshots`, and the capture hint hidden, so that irrelevant `Recording` controls do not clutter the form.
10. As a user who adds a `Recording` alongside a `Transcript` or clears the `Transcript`, I want `Screenshot interval` and `Max screenshots` restored immediately, so that I can configure `Screenshot` capture whenever a `Recording` is present.
11. As a user clicking `Reset`, I want the `Screenshots` tab hidden, the `Generate Meeting Notes` button re-enabled, and the capture sliders restored to their default visible state, so that the workspace returns cleanly to its initial state.

## Implementation Decisions

- **Layer boundary**: All changes live strictly in the UI and view presentation layers; pipeline, LLM, video, and MCP services remain untouched.
- **Header & theme persistence**: Move the `Provider` status chip into the title row alongside the theme toggle. Update the client-side theme toggle script to read/write the user's preference in `localStorage` on load and click, and update the toggle button label to reflect the active mode.
- **Dedicated `Screenshots` tab**: Relocate the `Screenshots` gallery inside the results tabs as a `Screenshots` tab item, style its container for dark mode without light-only backgrounds, and control the tab item's visibility from the `ResultsPaneState` / handler outputs so it is hidden when no `Screenshots` exist.
- **Compact share actions**: Constrain `Copy Markdown` and `Download HTML` to natural content width (`scale=0`) aligned to the right of the share actions row.
- **4-step progress card & button lock**:
  - Extend the pure stage-progress view helper to render a 4-step card (`Extracting Screenshots`, `Transcribing`, `Summarizing`, `Done`) with completed, active, and pending states plus elapsed time (skipping `Extracting Screenshots` and `Transcribing` visually or marking them appropriately for `Transcript`-only runs).
  - Update the generate-start handler to return an interactive-disabled update (`"Generating..."`) for the primary button, and restore it on completion, failure, or reset.
- **Context-sensitive capture sliders**: Wire input-change handlers for both `Recording` and `Transcript` inputs so that `Screenshot interval`, `Max screenshots`, and the capture hint are hidden if and only if a `Transcript` is present and `Recording` is absent, and visible otherwise.

## Testing Decisions

- **Existing seams only**: Test exclusively at the two established UI seams without introducing new test harnesses or touching pipeline/LLM tests:
  1. **Pure view-builder seam**: Test the 4-step progress card HTML helper and `ResultsPaneState` transitions (including `Screenshots` tab visibility) for `empty`, `generating`, `populated` (with and without `Screenshots`), and `error` phases.
  2. **App handler seam**: Test `MeetingNotesApp` handlers using fake services (following the prior art in the existing UI handler and Q&A lifecycle tests) to verify button locking/unlocking, `Screenshots` tab visibility updates, capture-slider visibility on `Transcript`-only vs `Recording` inputs, and `Reset` restoration.
- **Manual browser verification**: Theme persistence (`localStorage`), dark-mode gallery background, inline header alignment, and `Meeting Chat` viewport fit are verified manually in a browser across light, dark, and narrow viewports.

## Out of Scope

- Streaming token responses in `Meeting Chat`.
- Dynamic or content-derived starter questions in `Meeting Chat`.
- Clickable `Screenshot` or timestamp citations in `Meeting Chat`.
- Run history or reopening previous sessions.
- A `Cancel` button for in-flight generation.
- Changes to pipeline, LLM, video, or MCP behavior.

## Further Notes

- Vocabulary follows `GLOSSARY.md` (`Meeting Notes Helper`, `Recording`, `Transcript`, `Screenshot`, `Meeting Notes`, `Meeting Chat`, `Provider`).
- Builds on the baseline established in `specs/polished-ui.md`.
