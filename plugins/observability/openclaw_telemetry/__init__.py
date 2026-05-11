"""openclaw_telemetry — fire-and-log token usage telemetry for Hermes/Spark.

This plugin emits one OpenClaw ``token_usage`` event per Hermes LLM API call
using the ``post_api_request`` hook. It is intentionally optional and
fail-open: missing configuration, malformed usage, network failures, and server
errors are logged at warning/debug level but never raise into the dispatch path.

Configuration is environment-only so hooks do not read Hermes config per call:
  HERMES_OPENCLAW_TELEMETRY_ENV_FILE     optional path override
  HERMES_OPENCLAW_TELEMETRY_BASE_URL     preferred Spark Control endpoint
  HERMES_OPENCLAW_TELEMETRY_TOKEN        preferred Spark service token
  TELEMETRY_BASE_URL                     shared fallback endpoint
  TELEMETRY_TOKEN / TELEMETRY_WRITE_TOKEN fallback token names

Optional linkage fields:
  OPENCLAW_CARD_ID / HERMES_OPENCLAW_CARD_ID
  OPENCLAW_RUN_ID / HERMES_OPENCLAW_RUN_ID
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

from dotenv import load_dotenv

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_AGENT_ID = "spark"
_QUOTA_POOL = "openai_api"
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_QUEUE_SIZE = 100
_INIT_FAILED = object()
_CONFIG: object | Dict[str, Any] | None = None
_CONFIG_LOCK = threading.Lock()
_EVENT_QUEUE: queue.Queue[tuple[Dict[str, Any], Dict[str, Any]]] | None = None
_QUEUE_LOCK = threading.Lock()
_WORKER_STARTED = False
_ENV_FILE_LOADED = False


def _load_spark_env_file() -> None:
    """Load DEV Spark telemetry env from a separate, git-ignored file.

    The generated gateway systemd unit also imports this file when present, but
    existing DEV units may not have been regenerated with EnvironmentFile yet.
    Keep this runtime loading path load-bearing for tests, non-systemd launches,
    and already-installed services. The default lives under HERMES_HOME so token
    material never needs to be committed to the repo.
    """
    global _ENV_FILE_LOADED
    if _ENV_FILE_LOADED:
        return
    _ENV_FILE_LOADED = True

    configured = os.environ.get("HERMES_OPENCLAW_TELEMETRY_ENV_FILE", "").strip()
    env_path = Path(configured).expanduser() if configured else get_hermes_home() / "spark-telemetry.env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _first_env(*names: str) -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return ""


def reset_cache_for_tests() -> None:
    global _CONFIG, _EVENT_QUEUE, _WORKER_STARTED, _ENV_FILE_LOADED
    with _CONFIG_LOCK:
        _CONFIG = None
        _ENV_FILE_LOADED = False
    with _QUEUE_LOCK:
        _EVENT_QUEUE = None
        _WORKER_STARTED = False


def _flush_for_tests() -> None:
    queue_ref = _EVENT_QUEUE
    if queue_ref is not None:
        queue_ref.join()


def _get_config() -> Optional[Dict[str, Any]]:
    """Return cached runtime config, or None when telemetry is disabled.

    Missing env is cached as _INIT_FAILED so hot post_api_request hooks do not
    repeatedly re-read process env when the plugin is enabled but unconfigured.
    """
    global _CONFIG
    with _CONFIG_LOCK:
        if _CONFIG is _INIT_FAILED:
            return None
        if isinstance(_CONFIG, dict):
            return _CONFIG

        _load_spark_env_file()

        base_url = _first_env("HERMES_OPENCLAW_TELEMETRY_BASE_URL", "TELEMETRY_BASE_URL").rstrip("/")
        token = _first_env(
            "HERMES_OPENCLAW_TELEMETRY_TOKEN",
            "TELEMETRY_TOKEN",
            "TELEMETRY_WRITE_TOKEN",
        )
        if not (base_url and token):
            _CONFIG = _INIT_FAILED
            return None

        try:
            timeout_seconds = float(_env("HERMES_OPENCLAW_TELEMETRY_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)))
        except ValueError:
            logger.warning("Invalid HERMES_OPENCLAW_TELEMETRY_TIMEOUT_SECONDS; using default")
            timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
        if timeout_seconds <= 0:
            timeout_seconds = _DEFAULT_TIMEOUT_SECONDS

        _CONFIG = {
            "endpoint": f"{base_url}/v1/telemetry/token-usage",
            "token": token,
            "timeout_seconds": timeout_seconds,
        }
        return _CONFIG


def _usage_int(usage: Dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float) and value >= 0 and value.is_integer():
            return int(value)
    return 0


def _extract_cache_read_tokens(usage: Dict[str, Any]) -> int:
    direct = _usage_int(usage, "cache_read_tokens", "cacheReadTokens")
    if direct:
        return direct
    details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        return _usage_int(details, "cached_tokens", "cache_read_tokens", "cacheReadTokens")
    return 0


def _extract_cache_creation_tokens(usage: Dict[str, Any]) -> int:
    direct = _usage_int(
        usage,
        "cache_write_tokens",
        "cache_creation_input_tokens",
        "cache_creation_tokens",
        "cacheCreationTokens",
    )
    if direct:
        return direct
    details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        return _usage_int(
            details,
            "cache_write_tokens",
            "cache_creation_input_tokens",
            "cache_creation_tokens",
            "cacheCreationTokens",
        )
    return 0


def _duration_ms(kwargs: Dict[str, Any]) -> int:
    for key in ("duration_ms", "durationMs"):
        value = kwargs.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float) and value >= 0:
            return int(round(value))
    value = kwargs.get("api_duration")
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)) and value >= 0:
        return int(round(float(value) * 1000))
    return 0


def _event_id(task_id: str, session_id: str, api_call_count: Any, model: str) -> str:
    stable = f"spark:{task_id or ''}:{session_id or ''}:{api_call_count}:{model}"
    if stable == "spark::::":
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable))


def _optional_linkage(name: str) -> Optional[str]:
    value = _first_env(f"HERMES_OPENCLAW_{name}", f"OPENCLAW_{name}")
    return value or None


def _build_event(**kwargs: Any) -> Optional[Dict[str, Any]]:
    usage = kwargs.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    if input_tokens == 0 and output_tokens == 0:
        return None

    model = str(kwargs.get("response_model") or kwargs.get("model") or "").strip()
    if not model:
        return None

    return {
        "eventId": _event_id(
            str(kwargs.get("task_id") or ""),
            str(kwargs.get("session_id") or ""),
            kwargs.get("api_call_count", ""),
            model,
        ),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "agentId": _AGENT_ID,
        "model": model,
        "quotaPool": _QUOTA_POOL,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "cacheReadTokens": _extract_cache_read_tokens(usage),
        "cacheCreationTokens": _extract_cache_creation_tokens(usage),
        "runId": _optional_linkage("RUN_ID"),
        "cardId": _optional_linkage("CARD_ID"),
        "durationMs": _duration_ms(kwargs),
        "apiEquivalentCostUsd": 0,
    }


def _post_event(config: Dict[str, Any], event: Dict[str, Any]) -> None:
    body = json.dumps(event, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        config["endpoint"],
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with request.urlopen(req, timeout=float(config["timeout_seconds"])) as response:
        status = int(getattr(response, "status", 200))
        if status >= 400:
            raise RuntimeError(f"telemetry-service returned HTTP {status}")


def _emit_event(config: Dict[str, Any], event: Dict[str, Any]) -> None:
    try:
        _post_event(config, event)
        logger.info(
            "OpenClaw telemetry accepted event_id=%s model=%s input_tokens=%s output_tokens=%s card_id=%s run_id=%s",
            event["eventId"],
            event["model"],
            event["inputTokens"],
            event["outputTokens"],
            event.get("cardId"),
            event.get("runId"),
        )
    except (error.URLError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("OpenClaw telemetry write failed; continuing without blocking dispatch: %s", exc)
    except Exception as exc:  # pragma: no cover - belt-and-suspenders fail-open
        logger.warning("OpenClaw telemetry unexpected failure; continuing without blocking dispatch: %s", exc)


def _worker_loop(queue_ref: queue.Queue[tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
    while True:
        config, event = queue_ref.get()
        try:
            _emit_event(config, event)
        finally:
            queue_ref.task_done()


def _get_queue() -> queue.Queue[tuple[Dict[str, Any], Dict[str, Any]]]:
    global _EVENT_QUEUE, _WORKER_STARTED
    with _QUEUE_LOCK:
        if _EVENT_QUEUE is None:
            try:
                maxsize = int(_env("HERMES_OPENCLAW_TELEMETRY_QUEUE_SIZE", str(_DEFAULT_QUEUE_SIZE)))
            except ValueError:
                maxsize = _DEFAULT_QUEUE_SIZE
            _EVENT_QUEUE = queue.Queue(maxsize=max(1, maxsize))
        if not _WORKER_STARTED:
            worker = threading.Thread(
                target=_worker_loop,
                args=(_EVENT_QUEUE,),
                name="openclaw-telemetry-writer",
                daemon=True,
            )
            worker.start()
            _WORKER_STARTED = True
        return _EVENT_QUEUE


def _enqueue_event(config: Dict[str, Any], event: Dict[str, Any]) -> None:
    try:
        _get_queue().put_nowait((config, event))
    except queue.Full:
        logger.warning("OpenClaw telemetry queue full; dropping event_id=%s", event.get("eventId"))


def on_post_api_request(**kwargs: Any) -> None:
    """Queue token usage for one LLM API call without blocking dispatch."""
    config = _get_config()
    if config is None:
        return

    try:
        event = _build_event(**kwargs)
        if event is None:
            return
        _enqueue_event(config, event)
    except Exception as exc:  # pragma: no cover - fail-open at call site
        logger.warning("OpenClaw telemetry enqueue failed; continuing without blocking dispatch: %s", exc)


def register(ctx) -> None:
    ctx.register_hook("post_api_request", on_post_api_request)
