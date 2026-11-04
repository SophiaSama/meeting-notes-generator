import asyncio
import logging
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_RUNTIME_READY = False


def setup_runtime_environment() -> None:
    """Apply process-wide runtime setup once."""
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Keep Hugging Face inert: huggingface_hub is only a transitive Gradio
    # dependency. Must be set before Gradio/huggingface_hub are imported so the
    # hub never makes network calls. setdefault keeps any explicit override.
    for _hf_key, _hf_val in {
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DISABLE_TELEMETRY": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
    }.items():
        os.environ.setdefault(_hf_key, _hf_val)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            import asyncio.proactor_events as _proactor_events

            transport_cls = getattr(_proactor_events, "_ProactorBasePipeTransport", None)
            orig_cb = getattr(transport_cls, "_call_connection_lost", None) if transport_cls else None

            if callable(orig_cb) and not getattr(orig_cb, "_meeting_notes_patched", False):
                def safe_call_connection_lost(self, exc):
                    try:
                        return orig_cb(self, exc)
                    except ConnectionResetError as e:
                        if getattr(e, "winerror", None) == 10054:
                            return
                        raise

                setattr(safe_call_connection_lost, "_meeting_notes_patched", True)
                setattr(transport_cls, "_call_connection_lost", safe_call_connection_lost)
        except Exception:
            pass

    load_dotenv(dotenv_path=Path.cwd() / ".env")
    _RUNTIME_READY = True


def env_first(*keys: str, default: str = "") -> str:
    """Return the first non-empty environment variable from keys."""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def first_set_key(*keys: str) -> str | None:
    """Return first non-empty environment variable name from keys."""
    for key in keys:
        if os.getenv(key):
            return key
    return None


def mask_secret(value: str) -> str:
    """Return a masked representation safe for UI display."""
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


BASE_URL_KEYS = ("LLM_BASE_URL", "OPENAI_BASE_URL")
API_KEY_KEYS = ("LLM_API_KEY", "OPENAI_API_KEY")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ProxyDefaults:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = "none"

    @classmethod
    def from_env(cls) -> "ProxyDefaults":
        from .llm import REASONING_EFFORT_NONE, normalize_reasoning_effort

        reasoning_effort = (
            normalize_reasoning_effort(env_first("LLM_REASONING_EFFORT"))
            or REASONING_EFFORT_NONE
        )
        return cls(
            base_url=env_first(*BASE_URL_KEYS, default=DEFAULT_BASE_URL),
            api_key=env_first(*API_KEY_KEYS),
            model=env_first("LLM_MODEL", default=DEFAULT_MODEL),
            reasoning_effort=reasoning_effort,
        )

    def status_markdown(self) -> str:
        from .llm import LLMService

        base_url_src = first_set_key(*BASE_URL_KEYS) or "fallback"
        model_src = first_set_key("LLM_MODEL") or "fallback"
        api_key_src = first_set_key(*API_KEY_KEYS) or "empty"
        effort_src = first_set_key("LLM_REASONING_EFFORT") or "fallback"
        extra_header_names = ", ".join(LLMService.extra_headers()) or "(none)"

        return textwrap.dedent(
            f"""
            Loaded defaults:
            - Base URL: {self.base_url} (source: {base_url_src})
            - Model: {self.model} (source: {model_src})
            - API Key: {mask_secret(self.api_key)} (source: {api_key_src})
            - Extra headers: {extra_header_names} (source: LLM_EXTRA_HEADERS)
            - Reasoning effort: {self.reasoning_effort} (source: {effort_src})
            """
        ).strip()
