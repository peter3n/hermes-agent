"""Tests for the bundled observability/openclaw_telemetry plugin."""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from urllib import error

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "openclaw_telemetry"


@pytest.fixture(autouse=True)
def isolate_default_spark_env_file(tmp_path, monkeypatch):
    """Prevent real DEV telemetry env values from leaking into unit tests."""
    telemetry_keys = (
        "HERMES_OPENCLAW_TELEMETRY_ENV_FILE",
        "HERMES_OPENCLAW_TELEMETRY_BASE_URL",
        "HERMES_OPENCLAW_TELEMETRY_TOKEN",
        "HERMES_OPENCLAW_TELEMETRY_TIMEOUT_SECONDS",
        "HERMES_OPENCLAW_CARD_ID",
        "HERMES_OPENCLAW_RUN_ID",
        "TELEMETRY_BASE_URL",
        "TELEMETRY_TOKEN",
        "TELEMETRY_WRITE_TOKEN",
    )
    for key in telemetry_keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    yield
    # load_dotenv mutates os.environ directly, outside monkeypatch tracking.
    for key in telemetry_keys:
        monkeypatch.delenv(key, raising=False)


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "openclaw_telemetry"
        assert data["version"]
        assert set(data["hooks"]) == {"post_api_request"}
        assert data["requires_env"] == [
            "HERMES_OPENCLAW_TELEMETRY_BASE_URL",
            "HERMES_OPENCLAW_TELEMETRY_TOKEN",
        ]
        assert set(data["optional_env"]) == {
            "HERMES_OPENCLAW_TELEMETRY_ENV_FILE",
            "HERMES_OPENCLAW_TELEMETRY_TIMEOUT_SECONDS",
            "HERMES_OPENCLAW_CARD_ID",
            "HERMES_OPENCLAW_RUN_ID",
            "TELEMETRY_BASE_URL",
            "TELEMETRY_TOKEN",
            "TELEMETRY_WRITE_TOKEN",
        }


class TestRuntimeGate:
    def _fresh_plugin(self):
        mod_name = "plugins.observability.openclaw_telemetry"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_get_config_returns_none_without_credentials(self, monkeypatch):
        for key in (
            "HERMES_OPENCLAW_TELEMETRY_BASE_URL",
            "TELEMETRY_BASE_URL",
            "HERMES_OPENCLAW_TELEMETRY_TOKEN",
            "TELEMETRY_TOKEN",
            "TELEMETRY_WRITE_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)

        plugin = self._fresh_plugin()
        assert plugin._get_config() is None

    def test_get_config_caches_missing_env(self, monkeypatch):
        for key in (
            "HERMES_OPENCLAW_TELEMETRY_BASE_URL",
            "TELEMETRY_BASE_URL",
            "HERMES_OPENCLAW_TELEMETRY_TOKEN",
            "TELEMETRY_TOKEN",
            "TELEMETRY_WRITE_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)

        plugin = self._fresh_plugin()
        assert plugin._get_config() is None

        import os

        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key.startswith(("HERMES_OPENCLAW_", "TELEMETRY_")):
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)
        for _ in range(10):
            assert plugin._get_config() is None
        assert called["n"] == 0

    def test_get_config_prefers_spark_specific_token(self, monkeypatch):
        plugin = self._fresh_plugin()
        plugin.reset_cache_for_tests()
        monkeypatch.setenv("TELEMETRY_BASE_URL", "http://control.example:3001/")
        monkeypatch.setenv("TELEMETRY_TOKEN", "operator-token")
        monkeypatch.setenv("HERMES_OPENCLAW_TELEMETRY_TOKEN", "spark-token")

        config = plugin._get_config()

        assert config["endpoint"] == "http://control.example:3001/v1/telemetry/token-usage"
        assert config["token"] == "spark-token"

    def test_get_config_loads_default_spark_env_file(self, tmp_path, monkeypatch):
        plugin = self._fresh_plugin()
        plugin.reset_cache_for_tests()
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "spark-telemetry.env").write_text(
            "HERMES_OPENCLAW_TELEMETRY_BASE_URL=http://control.example:3001\n"
            "HERMES_OPENCLAW_TELEMETRY_TOKEN=spark-token\n",
            encoding="utf-8",
        )
        for key in (
            "HERMES_OPENCLAW_TELEMETRY_ENV_FILE",
            "HERMES_OPENCLAW_TELEMETRY_BASE_URL",
            "TELEMETRY_BASE_URL",
            "HERMES_OPENCLAW_TELEMETRY_TOKEN",
            "TELEMETRY_TOKEN",
            "TELEMETRY_WRITE_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = plugin._get_config()

        assert config["endpoint"] == "http://control.example:3001/v1/telemetry/token-usage"
        assert config["token"] == "spark-token"


class TestEventBuilder:
    def _fresh_plugin(self):
        mod_name = "plugins.observability.openclaw_telemetry"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_build_event_maps_usage_and_linkage(self, monkeypatch):
        plugin = self._fresh_plugin()
        monkeypatch.setenv("HERMES_OPENCLAW_CARD_ID", "card-123")
        monkeypatch.setenv("HERMES_OPENCLAW_RUN_ID", "run-456")

        event = plugin._build_event(
            task_id="task-1",
            session_id="session-1",
            api_call_count=2,
            response_model="openai/gpt-5.5",
            usage={
                "input_tokens": 123,
                "output_tokens": 45,
                "cache_read_tokens": 7,
                "cache_write_tokens": 3,
            },
            api_duration=0.891,
        )

        assert event["agentId"] == "spark"
        assert event["quotaPool"] == "openai_api"
        assert event["model"] == "openai/gpt-5.5"
        assert event["inputTokens"] == 123
        assert event["outputTokens"] == 45
        assert event["cacheReadTokens"] == 7
        assert event["cacheCreationTokens"] == 3
        assert event["durationMs"] == 891
        assert event["cardId"] == "card-123"
        assert event["runId"] == "run-456"

    def test_event_id_is_deterministic_for_stable_call_identity(self):
        plugin = self._fresh_plugin()
        kwargs = dict(
            task_id="task-1",
            session_id="session-1",
            api_call_count=2,
            response_model="openai/gpt-5.5",
            usage={"input_tokens": 1, "output_tokens": 2},
        )

        first = plugin._build_event(**kwargs)["eventId"]
        second = plugin._build_event(**kwargs)["eventId"]

        assert first == second

    @pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 0, "output_tokens": 0}])
    def test_build_event_skips_absent_or_zero_usage(self, usage):
        plugin = self._fresh_plugin()
        assert plugin._build_event(response_model="openai/gpt-5.5", usage=usage) is None


class TestHook:
    def _fresh_plugin(self):
        mod_name = "plugins.observability.openclaw_telemetry"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_post_api_request_posts_camel_case_json_without_raising(self, monkeypatch):
        plugin = self._fresh_plugin()
        plugin.reset_cache_for_tests()
        monkeypatch.setenv("TELEMETRY_BASE_URL", "http://control.example:3001")
        monkeypatch.setenv("HERMES_OPENCLAW_TELEMETRY_TOKEN", "spark-token")
        captured = {}

        class FakeResponse:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["auth"] = req.headers["Authorization"]
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse()

        monkeypatch.setattr(plugin.request, "urlopen", fake_urlopen)

        plugin.on_post_api_request(
            task_id="task-1",
            session_id="session-1",
            api_call_count=0,
            model="configured-model",
            response_model="openai/gpt-5.5",
            usage={"input_tokens": 10, "output_tokens": 5},
            api_duration=0.123,
        )
        plugin._flush_for_tests()

        assert captured["url"] == "http://control.example:3001/v1/telemetry/token-usage"
        assert captured["auth"] == "Bearer spark-token"
        assert captured["timeout"] == 5.0
        assert captured["body"]["agentId"] == "spark"
        assert captured["body"]["quotaPool"] == "openai_api"
        assert captured["body"]["inputTokens"] == 10
        assert captured["body"]["outputTokens"] == 5

    def test_post_api_request_fire_and_log_isolation(self, monkeypatch, caplog):
        plugin = self._fresh_plugin()
        plugin.reset_cache_for_tests()
        monkeypatch.setenv("TELEMETRY_BASE_URL", "http://control.example:3001")
        monkeypatch.setenv("HERMES_OPENCLAW_TELEMETRY_TOKEN", "spark-token")

        def fake_urlopen(req, timeout):
            raise error.URLError("boom")

        monkeypatch.setattr(plugin.request, "urlopen", fake_urlopen)

        plugin.on_post_api_request(
            task_id="task-1",
            session_id="session-1",
            api_call_count=0,
            response_model="openai/gpt-5.5",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        plugin._flush_for_tests()

        assert "continuing without blocking dispatch" in caplog.text
        assert "spark-token" not in caplog.text

    def test_post_api_request_does_not_wait_for_slow_http(self, monkeypatch):
        plugin = self._fresh_plugin()
        plugin.reset_cache_for_tests()
        monkeypatch.setenv("TELEMETRY_BASE_URL", "http://control.example:3001")
        monkeypatch.setenv("HERMES_OPENCLAW_TELEMETRY_TOKEN", "spark-token")

        def fake_urlopen(req, timeout):
            time.sleep(0.2)
            raise error.URLError("slow boom")

        monkeypatch.setattr(plugin.request, "urlopen", fake_urlopen)

        start = time.monotonic()
        plugin.on_post_api_request(
            task_id="task-1",
            session_id="session-1",
            api_call_count=0,
            response_model="openai/gpt-5.5",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        elapsed = time.monotonic() - start
        plugin._flush_for_tests()

        assert elapsed < 0.1
