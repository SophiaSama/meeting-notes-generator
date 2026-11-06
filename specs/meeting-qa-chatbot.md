# Meeting Q&A Chatbot Specification

## Problem Statement

After generating structured meeting notes from a recording or transcript, users often have follow-up questions that the static summary does not directly answer—such as specific details mentioned in passing during the discussion, exact wording or numbers from a captured slide screenshot, clarification on who raised a concern, or ad-hoc reformulations of action items. Currently, once the meeting notes are generated, the interaction ends: users must manually read through raw transcript artifacts or re-watch the video to answer follow-up questions about the meeting.

## Solution

Provide an interactive **Meeting Q&A Chatbot** directly inside the application once meeting notes generation completes. After a run produces structured meeting notes, the application retains the session's meeting context (generated meeting minutes, full or condensed transcript evidence, and frame OCR/timestamp metadata) and enables a multi-turn chat interface. Users can ask free-form questions about anything covered in the meeting and receive grounded answers using the same OpenAI-compatible LLM provider configuration and reasoning effort settings already configured in the UI. Resetting the session or generating notes for a new recording clears the prior chat history and binds the chatbot to the new meeting context.

## User Stories

1. As a meeting participant, I want a chat interface to become available after meeting notes are generated, so that I can immediately ask follow-up questions about the meeting without leaving the app.
2. As a meeting participant, I want the chatbot to be disabled or visually indicate when no meeting notes have been generated yet, so that I know I must generate notes before querying meeting content.
3. As a meeting participant, I want my questions to be answered using both the generated meeting minutes and the underlying transcript (or condensed transcript digest for long recordings), so that the chatbot can recall granular details that were summarized away in the high-level notes.
4. As a meeting participant who uploaded a video recording, I want the chatbot to have access to captured screenshot timestamps and extracted OCR text hints, so that I can ask about visual content shown on slides or shared screens during the meeting.
5. As a meeting participant, I want to ask multi-turn follow-up questions where the chatbot remembers earlier turns in the conversation, so that I can drill down on a topic naturally (e.g., "What did Bob say about the timeline?" followed by "Did anyone push back on that?").
6. As a meeting participant, I want the chatbot to cite or reference relevant timestamps, speakers, or screenshot numbers when that evidence is present in the transcript or OCR metadata, so that I can verify the answer against the meeting artifacts.
7. As a meeting participant, I want the chatbot to explicitly state when a question cannot be answered from the meeting content rather than hallucinating facts, so that I can trust the accuracy of its responses.
8. As a user configuring LLM settings, I want the chatbot to reuse the active LLM Provider Settings (Base URL, API Key, Model, and Reasoning Effort) from the application header, so that I do not have to configure credentials twice.
9. As a user experimenting with models, I want the ability to change the Model or Reasoning Effort dropdown between chat turns and have subsequent chat queries use the updated settings immediately, so that I can switch to a stronger reasoning model for complex questions.
10. As a user working with reasoning-capable models (such as Kimi, o1/o3/o4, or DeepSeek reasoners), I want the chatbot to display only the final visible answer and strip or suppress internal chain-of-thought reasoning, so that chat replies remain clean and readable.
11. As a user working with reasoning models that may hit token limits during thinking, I want the chatbot query pipeline to apply automatic output-limit retry and recovery behavior, so that chat turns do not fail silently with empty responses.
12. As a meeting participant, I want a "Clear Chat" action that resets only the conversation history while keeping the generated meeting notes and meeting context intact, so that I can start a fresh line of questioning on the same meeting.
13. As a meeting participant, I want clicking the global "Reset" button or running "Generate Meeting Notes" on a new upload to clear the previous chat history and update the active meeting context, so that answers from an old meeting never bleed into a new one.
14. As a multilingual user, I want the chatbot to default to the selected Output Language (or respond in the language of my question when appropriate), so that follow-up Q&A matches my language preference.
15. As a user, I want friendly, inline error feedback if an LLM connection error or invalid input occurs during a chat turn, so that a failed turn does not crash the session or lose my generated notes.

## Implementation Decisions

### Deep Module & Seam Placement
- **Extend `PipelineService` as the single primary deep module**: Rather than introducing a shallow standalone chat coordinator that duplicates prompt budgeting, token-limit recovery, reasoning-effort context management, and transcript condensation logic, `PipelineService` will own meeting Q&A context assembly and question answering.
- **Session Context Payload (`MeetingContext`)**:
  - When `PipelineService.run_pipeline` (and `run_pipeline_safe`) completes a run, it will construct and return (or store in per-session Gradio `gr.State`) a structured meeting context dictionary/dataclass containing:
    - `info`: recording/transcript metadata (`filename`, `duration_str`, `resolution`).
    - `markdown_notes`: the generated meeting minutes markdown body.
    - `transcript_evidence`: the full transcript text (or the map-reduce condensed transcript digest if the raw transcript exceeded the map-reduce trigger threshold, plus a fallback slice of raw transcript if within budget).
    - `transcript_source_label`: whether the evidence is the full transcript or a condensed digest.
    - `frames_metadata`: a lightweight, serializable list of frame summaries (`index`, `time_str`, `ocr_text`, `ocr_keywords`) without carrying raw PIL images or multi-megabyte base64 blobs in chat state.
    - `output_language`: the language selected during note generation.
    - `notes_style`: the style selected during note generation.

### Module Interfaces to Modify
- **`PipelineService`**:
  - Update `run_pipeline` / `run_pipeline_safe` to also produce the structured meeting context state (or provide a backward-compatible return signature / state helper so existing callers continue to work cleanly while the UI captures the session context).
  - Add `query_meeting(question, chat_history, meeting_context, base_url, api_key, model, output_language=None, reasoning_effort=None)` (and a UI-safe wrapper `query_meeting_safe`):
    - Validates that `meeting_context` is present and non-empty (raises a user-facing error or returns a polite prompt if notes have not been generated yet).
    - Validates that `question` is non-empty and that `base_url` and `model` are provided.
    - Enters `self.llm_service.reasoning_effort(reasoning_effort)` context manager so per-call reasoning effort is respected and isolated across concurrent Gradio sessions via `ContextVar`.
    - Builds a grounded system prompt instructing the model to act as a meeting Q&A assistant, prioritize the provided meeting transcript, OCR hints, and generated minutes, cite timestamps/speakers/screenshots where applicable, avoid fabricating unmentioned facts, and emit only the final answer without chain-of-thought.
    - Assembles the OpenAI-compatible `messages` array combining:
      1. Grounding context message (meeting metadata, generated minutes, frame OCR/timestamp index, and bounded transcript evidence sized to stay within `CHAT_REQUEST_SAFETY_RATIO` of the request byte budget).
      2. Prior conversation turns from `chat_history` (normalized to `{"role": "user" | "assistant", "content": ...}` pairs, trimming oldest turns if the cumulative payload approaches the request budget).
      3. The current user `question`.
    - Calls `_call_llm_with_length_retry` with stage `"meeting_qa"` so reasoning models get appropriate `max_tokens` budgets and automatic length-limit retry/fallback recovery.
    - Returns the assistant reply string (and updated chat history for the Gradio Chatbot adapter).
- **`MeetingNotesApp` (Gradio UI Adapter)**:
  - Add a `gr.State(value=None)` component to hold the active session's `meeting_context` in browser-session-isolated state (avoiding process-global mutable state across concurrent users).
  - Add a **"Chat with Meeting"** tab (next to "Markdown Notes", "HTML Preview", and "Download") or a dedicated Q&A panel directly below/beside the notes tabs:
    - `gr.Chatbot` component displaying the conversation history for the current meeting.
    - `gr.Textbox` input for the user's question (with Enter-to-submit and a "Send" `gr.Button`).
    - "Clear Chat" `gr.Button` to reset the conversation history without clearing the generated notes or `meeting_context`.
  - Wire `generate_btn.click` to populate the `meeting_context` state and reset the chatbot history when new notes are generated.
  - Wire `reset_btn.click` to clear the `meeting_context` state, chatbot history, and question input alongside existing outputs.

### Architectural & Prompt Budgeting Decisions
- **Text-Grounded Follow-up Queries (No Repeated Base64 Image Re-Uploads)**:
  - During note generation, screenshots are already processed via OCR (`_enrich_frames_with_document_model_ocr` / local OCR fallback) and integrated into the minutes. Re-sending megabytes of base64 JPEG payloads on every chat turn would add high latency and quickly exhaust request size budgets. Therefore, chat turns ground on the generated minutes, the transcript/digest, and the per-frame timestamp + OCR text summaries.
- **Session Isolation**:
  - Neither `PipelineService` nor `MeetingNotesApp` will store the active meeting's notes or chat history on `self`. All meeting context and chat history flow through `gr.State` and explicit method parameters on `PipelineService`, preserving concurrency safety and pure functional testability.

## Testing Decisions

### What Makes a Good Test
- Tests must exercise external behavior at the `PipelineService` interface seam, asserting on returned answers, updated conversation history, grounding payload contents sent to the LLM adapter, and error handling—never testing private helper methods or internal prompt formatting functions in isolation.

### Seams & Modules to Test
- **Primary Seam**: `PipelineService` (`query_meeting` / `query_meeting_safe` following `run_pipeline`), using the existing `FakeLLMService` and `FakeVideoService` test adapters at the network/ffmpeg boundary.
- **Scenarios to Cover at the Seam**:
  1. **End-to-end Generate-then-Query Flow**: Run `pipeline.run_pipeline` on a synthetic video or transcript to obtain the meeting context, then invoke `pipeline.query_meeting` with a user question and verify that the LLM receives the generated minutes, transcript text, and screenshot OCR/timestamp hints, and returns the assistant answer.
  2. **Multi-Turn Conversation History**: Pass prior user/assistant turns into `query_meeting` and verify that conversation history is preserved in order in the messages sent to `LLMService.call_llm` and appended in the returned history.
  3. **Missing Context Guardrail**: Calling `query_meeting` before notes are generated (`meeting_context=None` or empty) returns a clear user-facing message or raises `gr.Error` without invoking `LLMService.call_llm`.
  4. **Empty Question Guardrail**: Submitting a blank/whitespace question does not invoke `LLMService.call_llm` and leaves existing chat history intact.
  5. **Reasoning Effort & Output-Limit Recovery**: Passing `reasoning_effort` propagates through `LLMService.reasoning_effort` context, and if a reasoning model hits `LLMOutputLimitError` on the first chat attempt, `_call_llm_with_length_retry` recovers transparently.
  6. **Session Reset & Context Replacement**: Running `run_pipeline` a second time replaces the meeting context so subsequent `query_meeting` calls only include the new meeting's transcript and notes.

### Prior Art in Codebase
- End-to-end pipeline tests in `tests/test_e2e_pipeline.py` using `FakeLLMService`, `FakeVideoService`, `isolated_cwd`, and `synthetic_video` fixtures defined in `tests/conftest.py`.
- Reasoning effort and output-limit retry tests in `tests/test_llm_reasoning_control.py` and `tests/test_llm_service.py`.

## Out of Scope

- Cross-meeting search or querying across multiple historical `.minutes.md` files simultaneously (each chat session is scoped to the currently generated meeting).
- Persistent database storage of chat logs across browser reloads or server restarts.
- Vector database / embedding index (RAG) infrastructure—grounding uses the session's structured minutes, frame OCR index, and full/condensed transcript directly within the model's context budget.
- Re-sending raw base64 video frames on every chat turn (visual context is represented via the generated minutes and per-frame OCR + timestamp metadata).
- Voice/audio input for the chatbot.

## Further Notes

- Follow the repository's existing Kimi / reasoning model handling conventions documented in `docs/kimi-reasoning-response-handling.md` so that `reasoning_content` is never leaked into visible chatbot bubbles.
