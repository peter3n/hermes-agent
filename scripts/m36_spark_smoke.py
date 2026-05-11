#!/usr/bin/env python3
"""Emit one deterministic Spark OpenClaw telemetry row.

This script is intended for the M3.6 closure harness. Run it from the deployed
Hermes checkout on DEV, usually via CONTROL -> DEV SSH exec:

    cd /home/agent/.hermes/hermes-agent && \
      HERMES_HOME=/home/agent/.hermes \
      /home/agent/.hermes/hermes-agent/venv/bin/python scripts/m36_spark_smoke.py

It instantiates AIAgent directly for a one-turn, no-tool conversation so the
harness exercises the real provider path and the real post_api_request hook
without routing through Telegram, Mission Control dispatch, or gateway adapters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HERMES_HOME = "/home/agent/.hermes"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_PROVIDER = "openai-codex"
DEFAULT_CARD_ID = "m36-closure-spark"
DEFAULT_PROMPT = "Reply exactly: M3.6_SPARK_OK"


def _json_line(payload: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)


def _plugin_loaded() -> tuple[bool, str]:
    """Return whether the OpenClaw telemetry hook is registered."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    loaded = manager._plugins.get("observability/openclaw_telemetry") or manager._plugins.get(
        "openclaw_telemetry"
    )
    if loaded is None:
        return False, "observability/openclaw_telemetry was not discovered"
    if not loaded.enabled:
        return False, loaded.error or "observability/openclaw_telemetry is not enabled"
    hooks = getattr(loaded, "hooks_registered", []) or []
    if "post_api_request" not in hooks:
        return False, "observability/openclaw_telemetry did not register post_api_request"
    return True, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Spark no-tool LLM turn and flush OpenClaw telemetry."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HERMES_M36_SMOKE_MODEL", DEFAULT_MODEL),
        help=f"Model to use for the smoke call (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("HERMES_M36_SMOKE_PROVIDER", DEFAULT_PROVIDER),
        help=f"Provider to use for the smoke call (default: {DEFAULT_PROVIDER!r}).",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("HERMES_OPENCLAW_RUN_ID")
        or os.environ.get("OPENCLAW_RUN_ID")
        or f"m36-spark-{uuid.uuid4()}",
        help="Run ID to attach to the emitted telemetry row.",
    )
    parser.add_argument(
        "--card-id",
        default=os.environ.get("HERMES_OPENCLAW_CARD_ID")
        or os.environ.get("OPENCLAW_CARD_ID")
        or DEFAULT_CARD_ID,
        help=f"Card ID to attach to the emitted telemetry row (default: {DEFAULT_CARD_ID!r}).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt for the one-turn smoke call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("HERMES_M36_SMOKE_MAX_TOKENS", "16")),
        help="Maximum output tokens for the smoke call (default: 16).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("HERMES_HOME", DEFAULT_HERMES_HOME)
    os.environ["HERMES_OPENCLAW_RUN_ID"] = args.run_id
    os.environ["HERMES_OPENCLAW_CARD_ID"] = args.card_id

    session_id = f"m36_spark_{uuid.uuid4().hex}"
    task_id = f"m36_spark_{uuid.uuid4().hex}"

    try:
        from hermes_cli.plugins import discover_plugins
        from run_agent import AIAgent

        discover_plugins()

        ok, reason = _plugin_loaded()
        if not ok:
            _json_line(
                {
                    "ok": False,
                    "error": "openclaw_telemetry_not_loaded",
                    "detail": reason,
                    "runId": args.run_id,
                    "cardId": args.card_id,
                    "sessionId": session_id,
                    "taskId": task_id,
                },
                stream=sys.stderr,
            )
            return 2

        from plugins.observability import openclaw_telemetry

        if openclaw_telemetry._get_config() is None:
            _json_line(
                {
                    "ok": False,
                    "error": "openclaw_telemetry_not_configured",
                    "detail": "Missing telemetry base URL or token. Expected process env or HERMES_HOME/spark-telemetry.env.",
                    "hermesHome": os.environ.get("HERMES_HOME"),
                    "runId": args.run_id,
                    "cardId": args.card_id,
                    "sessionId": session_id,
                    "taskId": task_id,
                },
                stream=sys.stderr,
            )
            return 3

        agent = AIAgent(
            model=args.model,
            provider=args.provider,
            max_iterations=1,
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="m36-closure-harness",
            session_id=session_id,
            fallback_model=[],
            max_tokens=args.max_tokens,
        )

        result = agent.run_conversation(
            user_message=args.prompt,
            task_id=task_id,
        )

        # The telemetry worker posts asynchronously; drain before CONTROL queries.
        openclaw_telemetry._flush_for_tests()

        _json_line(
            {
                "ok": True,
                "runId": args.run_id,
                "cardId": args.card_id,
                "sessionId": session_id,
                "taskId": task_id,
                "model": args.model,
                "provider": args.provider,
                "finalResponse": result.get("final_response") if isinstance(result, dict) else result,
            }
        )
        return 0
    except Exception as exc:  # pragma: no cover - exercised by external harness failures.
        _json_line(
            {
                "ok": False,
                "error": exc.__class__.__name__,
                "detail": str(exc),
                "runId": args.run_id,
                "cardId": args.card_id,
                "sessionId": session_id,
                "taskId": task_id,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
