from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx
from openai import BadRequestError, OpenAI, UnprocessableEntityError

logger = logging.getLogger(__name__)

# Reasoning-effort levels selectable from the UI / MCP. "none" means: send no
# reasoning setting at all (also suppresses the env-based extra_body).
REASONING_EFFORT_NONE = "none"
REASONING_EFFORT_CHOICES: tuple[str, ...] = (
    REASONING_EFFORT_NONE,
    "minimal",
    "low",
    "medium",
    "high",
)

# Per-run reasoning effort. ``None`` = not specified by the caller (fall back to
# the env-configured extra_body); a value from REASONING_EFFORT_CHOICES otherwise.
# A ContextVar keeps concurrent Gradio sessions isolated from each other.
_reasoning_effort_ctx: ContextVar[str | None] = ContextVar(
    "meeting_notes_reasoning_effort", default=None
)

# Errors that indicate the provider/model rejected a request parameter.
_PARAM_REJECTION_ERRORS = (BadRequestError, UnprocessableEntityError)


def normalize_reasoning_effort(value: Any) -> str | None:
    """Normalise a UI/MCP effort value; ``None``/empty means "not specified"."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("off", "disabled"):
        return REASONING_EFFORT_NONE
    if text not in REASONING_EFFORT_CHOICES:
        logger.warning("Ignoring unknown reasoning effort %r.", value)
        return None
    return text


def build_reasoning_payload(effort: str | None) -> dict | None:
    """Return ``{"reasoning": {"effort": effort}}`` or ``None`` for none/unset."""
    effort = normalize_reasoning_effort(effort)
    if effort is None or effort == REASONING_EFFORT_NONE:
        return None
    return {"reasoning": {"effort": effort}}


@dataclass
class LLMResponse:
    """Normalised response from any LLM backend.

    ``content`` is always the visible answer text (never reasoning).
    ``reasoning`` captures chain-of-thought when the model returns it in a
    separate attribute (e.g. ``reasoning_content`` on Kimi / o1 models).
    """

    content: str
    reasoning: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


class _BatchRequestsFailedError(RuntimeError):
    """Raised when individual requests inside a batch failed."""


class LLMOutputLimitError(RuntimeError):
    """Raised when a model exhausts its output budget before visible content."""

    def __init__(
        self,
        *,
        model: str,
        requested_max_tokens: int,
        response: LLMResponse,
        stage: str,
    ):
        self.model = model
        self.requested_max_tokens = requested_max_tokens
        self.finish_reason = response.finish_reason
        self.input_tokens = response.input_tokens
        self.output_tokens = response.output_tokens
        self.reasoning_tokens = response.reasoning_tokens
        self.has_separate_reasoning = bool(response.reasoning)
        self.stage = stage
        super().__init__(
            f"Model '{model}' exhausted max_tokens before producing visible content "
            f"(stage={stage!r}, requested_max_tokens={requested_max_tokens}, "
            f"finish_reason={response.finish_reason!r}, input_tokens={response.input_tokens}, "
            f"output_tokens={response.output_tokens}, reasoning_tokens={response.reasoning_tokens}, "
            f"has_separate_reasoning={self.has_separate_reasoning})."
        )


class LLMService:
    """Wrapper around OpenAI-compatible chat, batch, and audio APIs."""

    BATCH_UPLOAD_HARD_LIMIT_BYTES = 200 * 1024 * 1024
    DEFAULT_BATCH_JSONL_SOFT_LIMIT_BYTES = 150 * 1024 * 1024
    DEFAULT_BATCH_POLL_INTERVAL_SECONDS = 5
    DEFAULT_BATCH_MAX_WAIT_SECONDS = 300
    DEFAULT_BATCH_COMPLETION_WINDOW = "24h"

    # Models known to separate reasoning from content via a dedicated attribute
    # (e.g. ``message.reasoning_content``).
    _REASONING_MODEL_PREFIXES = (
        "kimi",
        "o1",
        "o3",
        "o4",
        "deepseek-r1",
        "deepseek-reasoner",
    )

    DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
    DEFAULT_REASONING_REQUEST_TIMEOUT_SECONDS = 300.0

    def __init__(self) -> None:
        # (model_lower, payload_json) pairs the provider has rejected. Such
        # payloads are not sent again to that model for the rest of the process.
        self._rejected_reasoning_controls: set[tuple[str, str]] = set()

    @staticmethod
    @contextmanager
    def reasoning_effort(effort: Any) -> Iterator[None]:
        """Apply a reasoning effort to all LLM calls made inside the block."""
        token = _reasoning_effort_ctx.set(normalize_reasoning_effort(effort))
        try:
            yield
        finally:
            _reasoning_effort_ctx.reset(token)

    @staticmethod
    def current_reasoning_effort() -> str | None:
        return _reasoning_effort_ctx.get()

    @staticmethod
    def _rejection_key(model: str, payload: dict) -> tuple[str, str]:
        return ((model or "").lower(), json.dumps(payload, sort_keys=True))

    def _mark_reasoning_control_rejected(self, model: str, payload: dict, exc: Exception) -> None:
        rejected = getattr(self, "_rejected_reasoning_controls", None)
        if rejected is None:
            rejected = set()
            self._rejected_reasoning_controls = rejected
        key = self._rejection_key(model, payload)
        if key in rejected:
            return
        rejected.add(key)
        logger.info(
            "Model %s rejected reasoning setting %s; sending requests without it "
            "for this model from now on (%s).",
            model,
            key[1],
            str(exc)[:200],
        )

    @classmethod
    def _reasoning_model_prefixes(cls) -> tuple[str, ...]:
        raw_value = os.getenv("LLM_REASONING_MODEL_PREFIXES", "").strip()
        if not raw_value:
            return cls._REASONING_MODEL_PREFIXES
        prefixes = tuple(
            part.strip().lower() for part in raw_value.split(",") if part.strip()
        )
        return prefixes or cls._REASONING_MODEL_PREFIXES

    @classmethod
    def _is_reasoning_model(cls, model: str) -> bool:
        """Detect models that may spend output budget on hidden reasoning."""
        model_lower = (model or "").lower()
        return any(model_lower.startswith(prefix) for prefix in cls._reasoning_model_prefixes())

    @classmethod
    def _request_timeout_seconds(cls, model: str) -> float:
        raw_value = os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "").strip()
        if raw_value:
            try:
                return max(10.0, float(raw_value))
            except ValueError:
                pass
        if cls._is_reasoning_model(model):
            return cls.DEFAULT_REASONING_REQUEST_TIMEOUT_SECONDS
        return cls.DEFAULT_REQUEST_TIMEOUT_SECONDS

    @staticmethod
    def _configured_reasoning_control() -> dict | None:
        """Optional provider-specific reasoning control from the environment.

        Example values: ``{"thinking": {"type": "disabled"}}`` (Kimi K2.x) or
        ``{"reasoning_effort": "low"}``. Empty by default: nothing is sent.
        """
        raw_value = os.getenv("LLM_REASONING_CONTROL_EXTRA_BODY", "").strip()
        if not raw_value:
            return None
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            logger.warning("Ignoring LLM_REASONING_CONTROL_EXTRA_BODY: not valid JSON.")
            return None
        if not isinstance(value, dict) or not value:
            logger.warning("Ignoring LLM_REASONING_CONTROL_EXTRA_BODY: expected a JSON object.")
            return None
        return value

    def _reasoning_control_for(self, model: str) -> dict | None:
        """Resolve the reasoning extra_body for this call.

        1. Effort chosen via UI/MCP: "none" -> nothing; otherwise send
           ``{"reasoning": {"effort": X}}`` to every model.
        2. Not specified: env ``LLM_REASONING_CONTROL_EXTRA_BODY`` for
           reasoning-prefixed models only.
        Payloads previously rejected by the model are skipped.
        """
        effort = self.current_reasoning_effort()
        if effort is not None:
            payload = build_reasoning_payload(effort)
        elif self._is_reasoning_model(model):
            payload = self._configured_reasoning_control()
        else:
            payload = None
        if not payload:
            return None
        rejected = getattr(self, "_rejected_reasoning_controls", set())
        if self._rejection_key(model, payload) in rejected:
            return None
        return payload

    @staticmethod
    def extra_headers() -> dict[str, str]:
        """Optional provider-specific headers from ``LLM_EXTRA_HEADERS`` (JSON object)."""
        raw_value = os.getenv("LLM_EXTRA_HEADERS", "").strip()
        if not raw_value:
            return {}
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            logger.warning("Ignoring LLM_EXTRA_HEADERS: not valid JSON.")
            return {}
        if not isinstance(value, dict):
            logger.warning("Ignoring LLM_EXTRA_HEADERS: expected a JSON object.")
            return {}
        return {str(k): str(v) for k, v in value.items()}

    def build_client(
        self, base_url: str, api_key: str, *, timeout: float | None = None
    ) -> OpenAI:
        kwargs: dict[str, Any] = {
            "base_url": base_url.rstrip("/"),
            "api_key": api_key or "no-key",
            "default_headers": self.extra_headers() or None,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)

    @classmethod
    def _batch_jsonl_soft_limit_bytes(cls) -> int:
        raw_limit = os.getenv("LLM_BATCH_JSONL_SOFT_LIMIT_BYTES", "").strip()
        try:
            value = int(raw_limit) if raw_limit else cls.DEFAULT_BATCH_JSONL_SOFT_LIMIT_BYTES
        except ValueError:
            value = cls.DEFAULT_BATCH_JSONL_SOFT_LIMIT_BYTES
        return max(1_000_000, min(value, cls.BATCH_UPLOAD_HARD_LIMIT_BYTES - 1))

    @classmethod
    def _batch_poll_interval_seconds(cls) -> int:
        raw_interval = os.getenv("LLM_BATCH_POLL_INTERVAL_SECONDS", "").strip()
        try:
            value = int(raw_interval) if raw_interval else cls.DEFAULT_BATCH_POLL_INTERVAL_SECONDS
        except ValueError:
            value = cls.DEFAULT_BATCH_POLL_INTERVAL_SECONDS
        return max(1, value)

    @classmethod
    def _batch_max_wait_seconds(cls) -> int:
        raw_wait = os.getenv("LLM_BATCH_MAX_WAIT_SECONDS", "").strip()
        try:
            value = int(raw_wait) if raw_wait else cls.DEFAULT_BATCH_MAX_WAIT_SECONDS
        except ValueError:
            value = cls.DEFAULT_BATCH_MAX_WAIT_SECONDS
        return max(10, value)

    @classmethod
    def _batch_completion_window(cls) -> str:
        raw_window = os.getenv("LLM_BATCH_COMPLETION_WINDOW", "").strip()
        return raw_window or cls.DEFAULT_BATCH_COMPLETION_WINDOW

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_answer_from_message(message: Any) -> str:
        """Extract the visible answer text from a chat completion message.

        Handles both SDK objects (with ``content`` attribute) and plain dicts.
        Reasoning / thinking content is deliberately excluded — it is captured
        separately in ``LLMResponse.reasoning``.
        """
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = ""

        if isinstance(content, str):
            return content.strip()

        # Fallback: content-as-list (multimodal / structured responses)
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")
                if part_type in ("reasoning", "thinking"):
                    continue
                if part_type == "text":
                    annotations = part.get("annotations", [])
                    if annotations and any(
                        ann.get("type") in ("reasoning", "thinking")
                        for ann in annotations
                    ):
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
            if text_parts:
                return "\n".join(text_parts).strip()

        return ""

    @staticmethod
    def _get_reasoning_from_message(message: Any) -> str | None:
        """Extract reasoning/thinking content from a chat completion message.

        For Kimi / o1 models this lives in ``reasoning_content``.
        For other providers it may be serialised as ``reasoning`` or ``thinking``
        content parts.
        """
        reasoning = None

        if hasattr(message, "reasoning_content"):
            reasoning = getattr(message, "reasoning_content", None)
        elif isinstance(message, dict):
            reasoning = message.get("reasoning_content")

        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        # Fallback: look for reasoning/thinking content parts
        if hasattr(message, "content"):
            content = message.content  # type: ignore[union-attr]
        elif isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = ""
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("reasoning", "thinking"):
                    text = part.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "\n".join(parts)

        return None

    def _build_llm_response(
        self,
        response: Any,
    ) -> LLMResponse:
        """Build a normalised ``LLMResponse`` from an SDK chat completion."""
        raw: dict = {}
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                raw = dumped

        message_obj = None
        if hasattr(response, "choices"):
            choices = response.choices
            if choices and len(choices) > 0:
                first_choice = choices[0]
                if hasattr(first_choice, "message"):
                    message_obj = first_choice.message

        content = self._get_answer_from_message(message_obj) if message_obj is not None else ""
        reasoning = self._get_reasoning_from_message(message_obj) if message_obj is not None else None

        usage = raw.get("usage") if isinstance(raw, dict) else None
        input_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_details = (
            usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        )
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        reasoning_tokens = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        )
        finish_reason = None
        raw_choices = raw.get("choices") if isinstance(raw, dict) else None
        if isinstance(raw_choices, list) and raw_choices:
            first_choice = raw_choices[0]
            if isinstance(first_choice, dict):
                raw_finish_reason = first_choice.get("finish_reason")
                if isinstance(raw_finish_reason, str):
                    finish_reason = raw_finish_reason

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=finish_reason,
            raw=raw,
        )

    @staticmethod
    def _log_call_diagnostics(
        *,
        llm_response: LLMResponse,
        model: str,
        requested_max_tokens: int,
        stage: str,
        attempt: int,
        elapsed_seconds: float | None = None,
        reasoning_control_applied: bool = False,
    ) -> None:
        # Only the *length* of reasoning is logged — never its content.
        reasoning_chars = len(llm_response.reasoning) if llm_response.reasoning else 0
        logger.info(
            "LLM call diagnostics: stage=%s model=%s requested_max_tokens=%d "
            "finish_reason=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s "
            "has_separate_reasoning=%s attempt=%d has_visible_content=%s "
            "reasoning_chars=%d elapsed_s=%s reasoning_control=%s reasoning_effort=%s",
            stage,
            model,
            requested_max_tokens,
            llm_response.finish_reason,
            llm_response.input_tokens,
            llm_response.output_tokens,
            llm_response.reasoning_tokens,
            bool(llm_response.reasoning),
            attempt,
            bool(llm_response.content),
            reasoning_chars,
            f"{elapsed_seconds:.1f}" if elapsed_seconds is not None else "n/a",
            reasoning_control_applied,
            _reasoning_effort_ctx.get(),
        )

    def _validate_llm_response(
        self,
        llm_response: LLMResponse,
        model: str,
        *,
        requested_max_tokens: int = 0,
        stage: str = "unspecified",
    ) -> str:
        """Return the visible content or raise a descriptive error."""
        if llm_response.content:
            return llm_response.content

        raw = llm_response.raw if isinstance(llm_response.raw, dict) else {}
        finish_reason = llm_response.finish_reason or ""

        if finish_reason == "length":
            raise LLMOutputLimitError(
                model=model,
                requested_max_tokens=requested_max_tokens,
                response=llm_response,
                stage=stage,
            )

        detail = f"finish_reason={finish_reason!r}"

        usage = raw.get("usage")
        if usage is not None:
            detail += f", usage={usage}"
        raise RuntimeError(f"Model '{model}' returned an empty response ({detail}).")

    # ------------------------------------------------------------------
    # Chat API helpers (legacy, still used by internal filter logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_message(message: dict) -> str:
        """Legacy helper for batch output parsing (dict-based messages)."""
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")
                if part_type in ("reasoning", "thinking"):
                    continue
                if part_type == "text":
                    annotations = part.get("annotations", [])
                    if annotations and any(
                        ann.get("type") in ("reasoning", "thinking")
                        for ann in annotations
                    ):
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
            if text_parts:
                return "\n".join(text_parts).strip()

        return ""

    def _extract_chat_content_from_response_body(self, body: dict, model: str) -> str:
        """Parse a bare response dict (used by batch output parsing).

        Also checks for ``reasoning_content`` at the message level in the dict
        so that Kimi-style separated reasoning is logged but not returned as
        the answer.
        """
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"Model '{model}' returned malformed chat response body: missing choices."
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"Model '{model}' returned malformed chat choice.")

        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Model '{model}' returned malformed chat message.")

        usage = body.get("usage")
        completion_details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        response = LLMResponse(
            content=self._extract_text_from_message(message),
            reasoning=self._get_reasoning_from_message(message),
            input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
            reasoning_tokens=(
                completion_details.get("reasoning_tokens")
                if isinstance(completion_details, dict)
                else None
            ),
            finish_reason=first.get("finish_reason") if isinstance(first.get("finish_reason"), str) else None,
            raw=body,
        )
        return self._validate_llm_response(response, model)

    # ------------------------------------------------------------------
    # Batch infrastructure
    # ------------------------------------------------------------------

    def _poll_batch_until_complete(
        self,
        client: OpenAI,
        *,
        batch_id: str,
    ):
        poll_interval_seconds = self._batch_poll_interval_seconds()
        max_wait_seconds = self._batch_max_wait_seconds()
        waited = 0
        while True:
            batch = client.batches.retrieve(batch_id)
            status = getattr(batch, "status", "")
            if status == "completed":
                return batch
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(
                    f"Batch {batch_id} ended with status '{status}'."
                )
            if waited >= max_wait_seconds:
                raise TimeoutError(
                    "Batch did not complete in time. "
                    f"Waited {max_wait_seconds}s (status='{status}')."
                )
            time.sleep(poll_interval_seconds)
            waited += poll_interval_seconds

    def _render_batch_output_text(
        self,
        client: OpenAI,
        *,
        file_id: str,
    ) -> str:
        content_response = client.files.content(file_id)
        text_value = getattr(content_response, "text", None)
        if isinstance(text_value, str):
            return text_value
        if hasattr(content_response, "read"):
            raw_value = content_response.read()
            if isinstance(raw_value, bytes):
                return raw_value.decode("utf-8", errors="replace")
            if isinstance(raw_value, str):
                return raw_value
        raise RuntimeError(f"Unable to parse batch output file '{file_id}'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_llm(
        self,
        messages: list,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 1500,
        system_prompt: str | None = None,
        stage: str = "unspecified",
        attempt: int = 1,
    ) -> str:
        """Call the LLM and return the visible answer text.

        Reasoning content (when present) is logged but not included in the
        return value.
        """
        http_client = httpx.Client(
            transport=httpx.HTTPTransport(),
            timeout=self._request_timeout_seconds(model),
        )

        client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or "no-key",
            default_headers=self.extra_headers() or None,
            http_client=http_client,
        )

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": full_messages,
        }
        reasoning_control = self._reasoning_control_for(model)
        if reasoning_control:
            request_kwargs["extra_body"] = reasoning_control

        started = time.monotonic()
        try:
            response = client.chat.completions.create(**request_kwargs)
        except _PARAM_REJECTION_ERRORS as exc:
            if not reasoning_control:
                raise
            # The model may not support the optional reasoning setting (e.g.
            # non-reasoning models, or always-thinking models that cannot
            # disable thinking). Silently resend without it; only remember the
            # rejection once the plain request succeeds, so unrelated 400s
            # (e.g. oversized prompts) do not disable the setting.
            logger.debug(
                "Request with reasoning setting failed for model=%s stage=%s; "
                "retrying without it: %s",
                model,
                stage,
                exc,
            )
            rejected_payload = reasoning_control
            request_kwargs.pop("extra_body", None)
            reasoning_control = None
            response = client.chat.completions.create(**request_kwargs)
            self._mark_reasoning_control_rejected(model, rejected_payload, exc)
        elapsed = time.monotonic() - started

        llm_response = self._build_llm_response(response)
        self._log_call_diagnostics(
            llm_response=llm_response,
            model=model,
            requested_max_tokens=max_tokens,
            stage=stage,
            attempt=attempt,
            elapsed_seconds=elapsed,
            reasoning_control_applied=bool(reasoning_control),
        )
        return self._validate_llm_response(
            llm_response,
            model,
            requested_max_tokens=max_tokens,
            stage=stage,
        )

    def call_llm_batch_chat(
        self,
        *,
        messages_list: list[list[dict]],
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 1500,
        system_prompt: str | None = None,
        stage: str = "batch",
        attempt: int = 1,
    ) -> list[str]:
        if not messages_list:
            return []

        call_kwargs: dict[str, Any] = dict(
            messages_list=messages_list,
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stage=stage,
            attempt=attempt,
        )
        reasoning_control = self._reasoning_control_for(model)
        if not reasoning_control:
            return self._call_llm_batch_chat_once(reasoning_control=None, **call_kwargs)
        try:
            return self._call_llm_batch_chat_once(
                reasoning_control=reasoning_control, **call_kwargs
            )
        except (_BatchRequestsFailedError, *_PARAM_REJECTION_ERRORS) as exc:
            # Batch requests failed while carrying the optional reasoning
            # setting: silently rerun once without it. Remember the rejection
            # only if the plain batch succeeds.
            logger.debug(
                "Batch with reasoning setting failed for model=%s stage=%s; "
                "retrying without it: %s",
                model,
                stage,
                exc,
            )
            result = self._call_llm_batch_chat_once(reasoning_control=None, **call_kwargs)
            self._mark_reasoning_control_rejected(model, reasoning_control, exc)
            return result

    def _call_llm_batch_chat_once(
        self,
        *,
        messages_list: list[list[dict]],
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        system_prompt: str | None,
        stage: str,
        attempt: int,
        reasoning_control: dict | None,
    ) -> list[str]:
        requests: list[dict] = []
        ordered_custom_ids: list[str] = []
        reasoning_control = reasoning_control or {}
        for idx, messages in enumerate(messages_list, start=1):
            custom_id = f"mapreq-{idx:06d}"
            ordered_custom_ids.append(custom_id)
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        **reasoning_control,
                        "model": model,
                        "max_tokens": max_tokens,
                        "messages": full_messages,
                    },
                }
            )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".jsonl",
                delete=False,
            ) as jsonl_file:
                for request in requests:
                    jsonl_file.write(json.dumps(request, ensure_ascii=False) + "\n")
                temp_path = Path(jsonl_file.name)

            jsonl_size = temp_path.stat().st_size
            soft_limit = self._batch_jsonl_soft_limit_bytes()
            if jsonl_size >= self.BATCH_UPLOAD_HARD_LIMIT_BYTES:
                raise RuntimeError(
                    "Batch JSONL is too large for the files API "
                    f"({jsonl_size} bytes, hard limit {self.BATCH_UPLOAD_HARD_LIMIT_BYTES} bytes)."
                )
            if jsonl_size > soft_limit:
                raise RuntimeError(
                    "Batch JSONL exceeds configured safety threshold "
                    f"({jsonl_size} bytes > {soft_limit} bytes). "
                    "Reduce segment size or split the batch."
                )

            client = self.build_client(base_url, api_key)

            with temp_path.open("rb") as upload_file:
                file_response = client.files.create(
                    file=upload_file,
                    purpose="batch",
                )

            batch_response = client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window=self._batch_completion_window(),
            )

            batch = self._poll_batch_until_complete(
                client,
                batch_id=batch_response.id,
            )
            output_file_id = getattr(batch, "output_file_id", None)
            if not output_file_id:
                raise RuntimeError(
                    f"Batch {batch_response.id} completed without an output file ID."
                )

            output_text = self._render_batch_output_text(
                client,
                file_id=output_file_id,
            )

            responses_by_custom_id: dict[str, str] = {}
            failed_items: list[str] = []
            for line in output_text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                custom_id = item.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    failed_items.append("missing custom_id in batch output")
                    continue

                error = item.get("error")
                if error:
                    failed_items.append(f"{custom_id}: {error}")
                    continue

                response_payload = item.get("response")
                if not isinstance(response_payload, dict):
                    failed_items.append(f"{custom_id}: missing response payload")
                    continue
                status_code = response_payload.get("status_code")
                if isinstance(status_code, int) and status_code >= 400:
                    failed_items.append(f"{custom_id}: status_code={status_code}")
                    continue

                body = response_payload.get("body")
                if not isinstance(body, dict):
                    failed_items.append(f"{custom_id}: missing response body")
                    continue
                choices = body.get("choices")
                message = choices[0].get("message") if isinstance(choices, list) and choices else None
                usage = body.get("usage")
                completion_details = (
                    usage.get("completion_tokens_details") if isinstance(usage, dict) else None
                )
                batch_response = LLMResponse(
                    content=(
                        self._extract_text_from_message(message)
                        if isinstance(message, dict)
                        else ""
                    ),
                    reasoning=(
                        self._get_reasoning_from_message(message)
                        if isinstance(message, dict)
                        else None
                    ),
                    input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                    output_tokens=(
                        usage.get("completion_tokens") if isinstance(usage, dict) else None
                    ),
                    reasoning_tokens=(
                        completion_details.get("reasoning_tokens")
                        if isinstance(completion_details, dict)
                        else None
                    ),
                    finish_reason=(
                        choices[0].get("finish_reason")
                        if isinstance(choices, list)
                        and choices
                        and isinstance(choices[0], dict)
                        and isinstance(choices[0].get("finish_reason"), str)
                        else None
                    ),
                    raw=body,
                )
                self._log_call_diagnostics(
                    llm_response=batch_response,
                    model=model,
                    requested_max_tokens=max_tokens,
                    stage=stage,
                    attempt=attempt,
                    reasoning_control_applied=bool(reasoning_control),
                )
                responses_by_custom_id[custom_id] = self._validate_llm_response(
                    batch_response,
                    model,
                    requested_max_tokens=max_tokens,
                    stage=stage,
                )

            if failed_items:
                raise _BatchRequestsFailedError(
                    "Batch completed with failed requests. "
                    f"First failure: {failed_items[0]}"
                )

            missing = [cid for cid in ordered_custom_ids if cid not in responses_by_custom_id]
            if missing:
                raise RuntimeError(
                    "Batch output is incomplete. "
                    f"Missing {len(missing)} result(s), first missing custom_id={missing[0]}."
                )

            return [responses_by_custom_id[cid] for cid in ordered_custom_ids]
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def test_connection(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        """Quick ping to verify the LLM endpoint is reachable."""
        try:
            with self.reasoning_effort(reasoning_effort):
                reply = self.call_llm(
                    [{"role": "user", "content": "Reply with exactly: ok"}],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_tokens=512,
                    stage="test_connection",
                )
            return f"✅ Connected - model replied: \"{reply.strip()[:60]}\""
        except Exception as e:
            return f"❌ Failed: {e}"