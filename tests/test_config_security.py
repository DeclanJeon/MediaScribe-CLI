from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.config import (  # noqa: E402
    ALLOWED_PROVIDER_HOSTS,
    RuntimeConfig,
    build_runtime_config,
    load_environment,
    provider_candidates,
)
from mediascribe_cli.errors import ConfigError  # noqa: E402


def _clean_env(**overrides):
    """patch.dict(os.environ) with a fully cleared environment plus overrides."""
    return patch.dict(os.environ, overrides, clear=True)


class _RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **payload):
        self.events.append((event_type, payload))


class LoadEnvironmentTests(unittest.TestCase):
    def test_hostile_cwd_env_files_are_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("OPENROUTER_API_KEY=cwd-dotenv\n", encoding="utf-8")
            (root / ".env.local").write_text("OPENROUTER_API_KEY=cwd-dotenv-local\n", encoding="utf-8")
            home = root / "home"
            home.mkdir()
            previous_cwd = os.getcwd()
            try:
                os.chdir(root)
                with _clean_env(HOME=str(home)):
                    env = load_environment()
            finally:
                os.chdir(previous_cwd)
            self.assertNotIn("OPENROUTER_API_KEY", env)

    def test_explicit_env_file_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "secrets.env"
            env_file.write_text("OPENROUTER_API_KEY=from-file\nMEDIASCRIBE_MODEL=large\n", encoding="utf-8")
            home = root / "home"
            home.mkdir()
            with _clean_env(HOME=str(home)):
                env = load_environment(str(env_file))
            self.assertEqual(env["OPENROUTER_API_KEY"], "from-file")
            self.assertEqual(env["MEDIASCRIBE_MODEL"], "large")

    def test_mediascribe_env_file_variable_selects_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "custom.env"
            env_file.write_text("OPENCODE_API_KEY=via-var\n", encoding="utf-8")
            home = root / "home"
            home.mkdir()
            with _clean_env(HOME=str(home), MEDIASCRIBE_ENV_FILE=str(env_file)):
                env = load_environment()
            self.assertEqual(env["OPENCODE_API_KEY"], "via-var")

    def test_missing_explicit_env_file_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            home.mkdir()
            with _clean_env(HOME=str(home)):
                with self.assertRaises(ConfigError):
                    load_environment(str(Path(directory) / "absent.env"))

    def test_process_environment_wins_over_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "secrets.env"
            env_file.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
            home = root / "home"
            home.mkdir()
            with _clean_env(HOME=str(home), OPENROUTER_API_KEY="from-process"):
                env = load_environment(str(env_file))
            self.assertEqual(env["OPENROUTER_API_KEY"], "from-process")

    def test_user_config_env_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            user_env = home / ".config" / "mediascribe" / "env"
            user_env.parent.mkdir(parents=True)
            user_env.write_text("OPENCODE_API_KEY=user-level\n", encoding="utf-8")
            with _clean_env(HOME=str(home)):
                env = load_environment()
            self.assertEqual(env["OPENCODE_API_KEY"], "user-level")

    @unittest.skipIf(os.name == "nt", "POSIX symlink/mode semantics")
    def test_permissive_secret_file_warns_on_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "secrets.env"
            env_file.write_text("OPENROUTER_API_KEY=secret-value\n", encoding="utf-8")
            env_file.chmod(0o644)
            home = root / "home"
            home.mkdir()
            stderr = io.StringIO()
            with _clean_env(HOME=str(home)), contextlib.redirect_stderr(stderr):
                load_environment(str(env_file))
            self.assertIn("chmod", stderr.getvalue())

    @unittest.skipIf(os.name == "nt", "POSIX symlink/mode semantics")
    def test_restrictive_secret_file_does_not_warn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "secrets.env"
            env_file.write_text("OPENROUTER_API_KEY=secret-value\n", encoding="utf-8")
            env_file.chmod(0o600)
            home = root / "home"
            home.mkdir()
            stderr = io.StringIO()
            with _clean_env(HOME=str(home)), contextlib.redirect_stderr(stderr):
                load_environment(str(env_file))
            self.assertEqual(stderr.getvalue(), "")

    @unittest.skipIf(os.name == "nt", "POSIX symlink/mode semantics")
    def test_permissive_non_secret_file_does_not_warn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / "settings.env"
            env_file.write_text("MEDIASCRIBE_MODEL=small\n", encoding="utf-8")
            env_file.chmod(0o644)
            home = root / "home"
            home.mkdir()
            stderr = io.StringIO()
            with _clean_env(HOME=str(home)), contextlib.redirect_stderr(stderr):
                load_environment(str(env_file))
            self.assertEqual(stderr.getvalue(), "")


class ProviderEndpointTests(unittest.TestCase):
    def _env(self, **overrides):
        env = {
            "OPENROUTER_API_KEY": "or-key",
            "OPENCODE_API_KEY": "oc-key",
        }
        env.update(overrides)
        return env

    def test_config_error_exit_code_is_two(self) -> None:
        self.assertEqual(ConfigError.exit_code, 2)
        self.assertEqual(ConfigError("x").exit_code, 2)

    def test_default_candidates_use_allowlisted_https_hosts(self) -> None:
        candidates = provider_candidates(self._env())
        self.assertEqual([c.id for c in candidates], ["openrouter", "opencode"])
        for candidate in candidates:
            self.assertTrue(candidate.base_url.startswith("https://"))
            host = candidate.base_url.split("/")[2]
            self.assertIn(host, ALLOWED_PROVIDER_HOSTS[candidate.id])

    def test_provider_order_is_preserved(self) -> None:
        candidates = provider_candidates(self._env())
        self.assertEqual([c.id for c in candidates], ["openrouter", "opencode"])
        only_opencode = provider_candidates(self._env(), provider="opencode")
        self.assertEqual([c.id for c in only_opencode], ["opencode"])

    def test_http_base_url_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            provider_candidates(self._env(OPENROUTER_BASE_URL="http://openrouter.ai/api/v1"))

    def test_evil_http_endpoint_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            provider_candidates(self._env(OPENROUTER_BASE_URL="http://evil.example"))

    def test_unknown_https_host_is_rejected_without_opt_in(self) -> None:
        with self.assertRaises(ConfigError):
            provider_candidates(self._env(OPENROUTER_BASE_URL="https://proxy.internal.example/v1"))

    def test_malformed_base_url_is_rejected(self) -> None:
        for bad in ("not-a-url", "ftp://openrouter.ai", "https://", "https://openrouter.ai:badport"):
            with self.assertRaises(ConfigError, msg=bad):
                provider_candidates(self._env(OPENROUTER_BASE_URL=bad))

    def test_loopback_http_is_allowed(self) -> None:
        for url in ("http://127.0.0.1:8080/v1", "http://localhost:9000", "http://[::1]:11434"):
            candidates = provider_candidates(self._env(OPENROUTER_BASE_URL=url))
            self.assertEqual(candidates[0].base_url, url.rstrip("/"))

    def test_custom_https_endpoint_requires_opt_in(self) -> None:
        env = self._env(OPENROUTER_BASE_URL="https://gateway.corp.example/v1")
        with self.assertRaises(ConfigError):
            provider_candidates(env)
        emitter = _RecordingEmitter()
        candidates = provider_candidates(env, allow_custom_endpoint=True, emitter=emitter)
        self.assertEqual(candidates[0].base_url, "https://gateway.corp.example/v1")
        self.assertEqual(emitter.events[0][0], "config_warning")
        self.assertNotIn("or-key", str(emitter.events))

    def test_custom_endpoint_opt_in_via_environment(self) -> None:
        env = self._env(
            OPENROUTER_BASE_URL="https://gateway.corp.example/v1",
            MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT="1",
        )
        candidates = provider_candidates(env)
        self.assertEqual(candidates[0].base_url, "https://gateway.corp.example/v1")

    def test_opt_in_does_not_allow_non_loopback_http(self) -> None:
        env = self._env(OPENROUTER_BASE_URL="http://gateway.corp.example/v1")
        with self.assertRaises(ConfigError):
            provider_candidates(env, allow_custom_endpoint=True)


class RuntimeConfigTests(unittest.TestCase):
    def test_new_fields_default(self) -> None:
        config = RuntimeConfig()
        self.assertEqual(config.env_file, "")
        self.assertFalse(config.allow_custom_endpoint)
        self.assertTrue(config.resume)
        self.assertFalse(config.follow_symlinks)
        self.assertEqual(config.summary_chunk_chars, 24_000)
        self.assertFalse(config.events_full_text)

    def test_new_fields_from_environment(self) -> None:
        env = {
            "MEDIASCRIBE_ENV_FILE": "/tmp/x.env",
            "MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT": "1",
            "MEDIASCRIBE_RESUME": "0",
            "MEDIASCRIBE_FOLLOW_SYMLINKS": "yes",
            "MEDIASCRIBE_SUMMARY_CHUNK_CHARS": "48000",
            "MEDIASCRIBE_EVENTS_FULL_TEXT": "true",
        }
        config = build_runtime_config(types.SimpleNamespace(), environment=env)
        self.assertEqual(config.env_file, "/tmp/x.env")
        self.assertTrue(config.allow_custom_endpoint)
        self.assertFalse(config.resume)
        self.assertTrue(config.follow_symlinks)
        self.assertEqual(config.summary_chunk_chars, 48_000)
        self.assertTrue(config.events_full_text)

    def test_args_override_environment(self) -> None:
        args = types.SimpleNamespace(
            env_file="/cli.env",
            allow_custom_endpoint=True,
            no_resume=True,
            follow_symlinks=True,
            summary_chunk_chars=8000,
            events_full_text=True,
        )
        config = build_runtime_config(args, environment={})
        self.assertEqual(config.env_file, "/cli.env")
        self.assertTrue(config.allow_custom_endpoint)
        self.assertFalse(config.resume)
        self.assertTrue(config.follow_symlinks)
        self.assertEqual(config.summary_chunk_chars, 8000)
        self.assertTrue(config.events_full_text)


if __name__ == "__main__":
    unittest.main()
