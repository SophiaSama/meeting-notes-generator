## Problem Statement

A non-technical user opening Meeting Notes Helper faces a crowded, utilitarian page. The Provider settings are always open and push the upload area down. Generation takes minutes and shows only a generic progress bar. Raw Markdown is shown before the readable Meeting Notes. Before the first run the results area is blank, and failures surface as raw text. The product is named inconsistently ("Agent" vs "Helper"), the look is default Gradio with a hard-coded light-only hint box, and narrow windows are hard to navigate.

## Solution

A polished, guided single-session experience. The page uses the full window width (already delivered) with a deliberate theme that supports dark mode. The Provider is shown as a compact status chip and is configured once. The main flow (add Recording or Transcript, generate, read Meeting Notes, ask Meeting Chat) is uncluttered. Progress is shown as named stages with elapsed time. Results open on the rendered Meeting Notes with copy and download buttons. Empty states and errors are friendly and actionable. Meeting Chat offers starter questions.

## User Stories

1. As a non-technical user, I want the Provider settings collapsed once configured, so that the upload area is the first thing I see.
2. As a user with no API key configured, I want the Provider settings to open automatically and be highlighted, so that I know what to fix first.
3. As a user, I want a status chip showing "Connected: <model>" or "Not configured", so that I can see the Provider state at a glance.
4. As a user, I want the Provider chip to update after Test Connection, so that I trust it reflects reality.
5. As a user, I want to see named stages (Extracting Screenshots, Transcribing, Summarizing, Done) while generating, so that I know the app is working and how far along it is.
6. As a user, I want an elapsed-time indicator during generation, so that long runs feel predictable.
7. As a user, I want Meeting Notes to open rendered, so that I can read them immediately.
8. As a user, I want a Copy Markdown button and a Download HTML button beside the Meeting Notes, so that sharing takes one click.
9. As a power user, I want raw Markdown still available as a secondary tab, so that I can inspect or edit it.
10. As a first-time user, I want a friendly empty state ("Add a Recording or Transcript to begin"), so that I know what to do.
11. As a user, I want errors shown as inline dismissible banners naming the likely cause and fix (invalid key, ffmpeg missing, file too large), so that I can recover without reading logs.
12. As a user, I want starter questions in Meeting Chat ("What were the action items?", "Who owns what?"), so that I can start without thinking of a question.
13. As a user, I want clicking a starter question to fill the input box, so that I can edit it before sending.
14. As a user, I want Meeting Chat starters hidden or disabled until Meeting Notes exist, so that I am not misled.
15. As a user, I want any citations (timestamps, speakers, Screenshot numbers) to appear as plain text in answers, so that I can verify them.
16. As a user, I want the product consistently called "Meeting Notes Helper" and the inputs and outputs called Recording, Transcript, and Meeting Notes, so that wording is clear.
17. As a user in dark mode, I want all text, including the capture hint, to be readable, so that nothing disappears.
18. As a user with a narrow window, I want columns to stack and Generate to stay reachable, so that I do not scroll to find it.
19. As a keyboard user, I want visible focus and labelled controls, so that I can operate the app without a mouse.
20. As a user, I want Reset to restore the empty state, clear the chip stages, errors, and chat, so that I can start over cleanly.

## Implementation Decisions

- Single-session only; no run history and no new domain concepts beyond the glossary.
- Theme: one accent color on a `gr.themes` base, with dark mode support. Custom CSS is kept minimal and uses theme variables (the capture hint box stops hard-coding colors).
- Extract pure view-builder helpers that return HTML strings: Provider status chip, stage progress, and error banner (cause plus suggested fix). The handlers compose them.
- Provider chip and accordion state are derived from existing defaults and the Test Connection result; the accordion opens when no key is configured.
- Stage progress is driven by progress callbacks the pipeline already emits; stage names are mapped in the view layer. No pipeline changes. Elapsed time is computed from run start.
- Results pane has three states: empty, generating, populated. Rendered Meeting Notes are the default view; Markdown and Download are secondary.
- Error classification is a small mapping from known failure messages to banner cause and fix; unknown errors show the message with a generic hint.
- Chat starter chips prefill the input only; they never auto-send. No structured citation UI because the Q&A service returns citations only as prompt-driven text.
- Responsive: columns stack below about 900px.
- Delivery order, one reviewable change each: (1) theme and terminology, (2) Provider chip and collapse, (3) results pane, (4) stage progress, (5) error banners, (6) chat chips.

## Testing Decisions

- Test only external behavior at two seams: `MeetingNotesApp` handler functions (returned Gradio updates for chip, visibility, stage text, banner, and chip prefill) and the pure view-builder helpers.
- Theme, CSS, and responsive stacking are verified manually in a browser (light, dark, narrow), not unit-tested.
- Prior art: `test_meeting_qa_chatbot.py` and `test_meeting_qa_lifecycle.py` already exercise the app handlers with fake services.
- Pipeline, LLM, and video services are not touched and need no new tests.

## Out of Scope

- Run history or reopening past runs.
- A Cancel button for generation.
- Structured, clickable citations in Meeting Chat.
- Changes to pipeline, LLM, video, or MCP behavior.
- Mobile-specific redesign beyond stacking.

## Further Notes

- Vocabulary follows `GLOSSARY.md`. No ADRs were needed.
- The full-width layout fix (max-width removal, taller gallery, `55vh` chat height) is already in place and is the baseline for this work.
- A prior spec for the chat feature is in `specs/meeting-qa-chatbot.md`.
