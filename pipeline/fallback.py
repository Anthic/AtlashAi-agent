"""
pipeline/fallback.py
────────────────────────────────────────────────────────────
Resilient Multi-Provider Fallback Cascade with Execution Metadata.
Zero-failure guarantee across Groq, Gemini, Mistral, and OpenRouter.
"""
import time
import logging
import threading
from typing import List, Dict, Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

from pipeline.model import get_llm

log = logging.getLogger(__name__)

# Fallback Cascade Sequences
DEFAULT_WORKER_CASCADE = [
    "worker-groq",
    "worker-gemini",
    "worker-mistral",
    "worker-openrouter",
]
DEFAULT_MASTER_CASCADE = [
    "master-mistral",
    "master-gemini",
    "master-openrouter",
]

# Per-call timeout in seconds. Prevents a hung/slow provider from stalling
# the whole cascade (llm.invoke() has no timeout by default).
DEFAULT_CALL_TIMEOUT_SEC = 20

# How long (seconds) to skip a provider after it fails, so a cascade doesn't
# keep re-trying a provider that's mid rate-limit-cooldown on every request.
COOLDOWN_SEC = 60

_cooldown_lock = threading.Lock()
_cooldown_until: Dict[str, float] = {}  # model_key -> unix timestamp


class CascadeExecutionResult:
    """Holds response content and execution performance metadata."""

    def __init__(self, content: str, provider_used: str, duration_sec: float, token_usage: Dict[str, int]):
        self.content = content
        self.provider_used = provider_used
        self.duration_sec = duration_sec
        self.token_usage = token_usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provider_used": self.provider_used,
            "duration_sec": round(self.duration_sec, 3),
            "token_usage": self.token_usage,
        }


def _is_on_cooldown(model_key: str) -> bool:
    with _cooldown_lock:
        until = _cooldown_until.get(model_key)
        return until is not None and time.time() < until


def _mark_cooldown(model_key: str) -> None:
    with _cooldown_lock:
        _cooldown_until[model_key] = time.time() + COOLDOWN_SEC


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """
    Provider-agnostic token usage extraction.
    LangChain standardizes usage onto `response.usage_metadata` (input_tokens/
    output_tokens/total_tokens) across providers. Fall back to the raw
    `response_metadata` dict (OpenAI-style: prompt_tokens/completion_tokens)
    for providers/older versions that don't populate usage_metadata.
    """
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta:
        token_usage["prompt_tokens"] = usage_meta.get("input_tokens", 0)
        token_usage["completion_tokens"] = usage_meta.get("output_tokens", 0)
        token_usage["total_tokens"] = usage_meta.get("total_tokens", 0)
        return token_usage

    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata:
        usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if usage:
            token_usage["prompt_tokens"] = usage.get("prompt_tokens", 0)
            token_usage["completion_tokens"] = usage.get("completion_tokens", 0)
            token_usage["total_tokens"] = usage.get("total_tokens", 0)

    return token_usage


def execute_with_fallback(
    messages_or_prompt: Any,
    tier: str = "worker",
    custom_cascade: Optional[List[str]] = None,
    stop: Optional[List[str]] = None,
    call_timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC,
) -> CascadeExecutionResult:
    """
    Executes an LLM call across a prioritized fallback chain.
    If the primary model fails (timeout, rate limit, error), it fails over seamlessly.

    If every model in the cascade is currently on cooldown from a recent
    failure, only the single model closest to recovery is probed (rather
    than bypassing cooldown for the whole cascade), to avoid re-hammering
    every provider at once during a shared outage or rate-limit burst.
    """
    if custom_cascade is None and tier not in ("worker", "master"):
        raise ValueError(f"Invalid tier: {tier!r}. Must be 'worker' or 'master'.")

    cascade = custom_cascade or (
        DEFAULT_WORKER_CASCADE if tier == "worker" else DEFAULT_MASTER_CASCADE
    )

    errors = []
    start_time = time.time()

    # ── Cooldown selection ────────────────────────────────────────────────
    # Normally skip any model still in cooldown. But if EVERY model in the
    # cascade is currently on cooldown (e.g. a shared outage or a burst of
    # rate limits), don't hammer all of them again on every incoming
    # request — that would turn one outage into a thundering herd against
    # every provider simultaneously, each paying the full call_timeout_sec.
    #
    # Instead, probe just the ONE model whose cooldown will expire soonest.
    # This still lets the system self-recover as soon as any provider comes
    # back, without re-trying every provider in full on every request.
    available_keys = [k for k in cascade if not _is_on_cooldown(k)]

    if available_keys:
        candidate_keys = available_keys
        skipped_keys = [k for k in cascade if k not in available_keys]
    else:
        with _cooldown_lock:
            probe_key = min(cascade, key=lambda k: _cooldown_until.get(k, 0))
        log.warning(
            "All cascade models are on cooldown; probing only [%s] "
            "(soonest to recover) instead of bypassing cooldown entirely.",
            probe_key,
        )
        candidate_keys = [probe_key]
        skipped_keys = [k for k in cascade if k != probe_key]

    for model_key in skipped_keys:
        log.info("Skipping [%s]: still in cooldown after a recent failure.", model_key)
        errors.append(f"{model_key}: skipped (cooldown)")

    for model_key in candidate_keys:
        try:
            log.info("Attempting LLM call on provider tier: %s", model_key)
            llm: BaseChatModel = get_llm(model_key)
            call_start = time.time()

            response = llm.invoke(
                messages_or_prompt,
                stop=stop,
                config={"timeout": call_timeout_sec},
            )
            call_duration = time.time() - call_start

            token_usage = _extract_token_usage(response)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join(part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content)
            log.info("Success with [%s] in %.2fs", model_key, call_duration)
            return CascadeExecutionResult(
                content=content,
                provider_used=model_key,
                duration_sec=time.time() - start_time,
                token_usage=token_usage,
            )

        except Exception as exc:
            log.warning(
                "Provider [%s] failed: %s. Initiating failover...",
                model_key, exc, exc_info=True,
            )
            _mark_cooldown(model_key)
            errors.append(f"{model_key}: {exc}")

            try:
                from pipeline.alert_watcher import notify_model_fallback
                curr_idx = candidate_keys.index(model_key)
                next_model = candidate_keys[curr_idx + 1] if curr_idx + 1 < len(candidate_keys) else "None (Exhausted)"
                notify_model_fallback(
                    failed_model=model_key,
                    next_model=next_model,
                    reason=str(exc),
                    agent_name=f"LLM Cascade ({tier})"
                )
            except Exception:
                pass
            continue

    total_duration = time.time() - start_time
    error_summary = " | ".join(errors)
    log.critical("All fallback cascade providers failed! Errors: %s", error_summary)

    try:
        from pipeline.alert_watcher import notify_pipeline_error
        notify_pipeline_error(
            stage=f"LLM Cascade ({tier}) - All Providers Failed",
            topic="LLM Inference",
            error_msg=error_summary
        )
    except Exception:
        pass

    raise RuntimeError(
        f"All LLM fallback providers exhausted ({total_duration:.2f}s). Root causes: {error_summary}"
    )