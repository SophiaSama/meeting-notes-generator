# Meeting Notes Helper

A tool that turns a meeting Recording or Transcript into structured Meeting Notes and lets the user ask questions about the meeting.

## Language

**Meeting Notes Helper**:
The product name, used in the UI title, README, and docs.
_Avoid_: Meeting Notes Agent, Meeting Notes Generator

**Recording**:
A video file of a meeting supplied by the user.
_Avoid_: Video, upload, media

**Transcript**:
The text of what was said in a meeting, either supplied by the user or produced from a Recording.
_Avoid_: Captions, script

**Screenshot**:
A frame captured from a Recording that shows presentation or shared-screen content.
_Avoid_: Frame, slide, image

**Meeting Notes**:
The structured minutes generated from a Recording or Transcript, available as rendered view, Markdown, and downloadable HTML.
_Avoid_: Minutes, summary, report, output

**Notes Style**:
The chosen shape of the Meeting Notes (e.g. Executive Summary, Action Items Focus).
_Avoid_: Format, template

**Meeting Chat**:
The question-and-answer conversation grounded in one meeting's Meeting Notes and Transcript.
_Avoid_: Chatbot, Q&A bot

**Provider**:
The configured LLM service (endpoint, key, model, reasoning effort) used to produce Meeting Notes and answer in Meeting Chat.
_Avoid_: Backend, LLM settings
