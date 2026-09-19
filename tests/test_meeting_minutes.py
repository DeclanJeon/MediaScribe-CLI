from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mediascribe_cli.chunking import (  # noqa: E402
    MAX_CHUNK_CHARS,
    MAX_ERROR_BODY_BYTES,
    MAX_RESPONSE_BYTES,
    chunk_digest,
    chunk_transcript,
)
from mediascribe_cli.errors import ConfigError, ProviderError  # noqa: E402
from mediascribe_cli.meeting_minutes import (  # noqa: E402
    MAX_REQUEST_BODY_BYTES,
    GeneratedMinutes,
    _NoRedirectHandler,
    generate_body_summary,
    generate_meeting_minutes,
)


def _http_error(code: int, body: bytes = b"") -> HTTPError:
    return HTTPError("https://provider.invalid/chat/completions", code, "err", {}, BytesIO(body))


def _payload(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}, ensure_ascii=False).encode("utf-8")


class FakeResponse:
    status = 200

    def __init__(self, text: str = "## 핵심 요약\n- 일정 논의", raw: bytes | None = None):
        self._raw = raw if raw is not None else _payload(text)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1):
        if size is None or size < 0:
            return self._raw
        return self._raw[:size]


class DictCheckpoint:
    """Minimal SummaryCheckpoint implementation backed by a dict."""

    def __init__(self, initial: dict[str, str] | None = None):
        self.store = dict(initial or {})
        self.get_calls: list[str] = []
        self.put_calls: list[str] = []

    def get(self, digest: str) -> str | None:
        self.get_calls.append(digest)
        return self.store.get(digest)

    def put(self, digest: str, text: str) -> None:
        self.put_calls.append(digest)
        self.store[digest] = text


class RecordingEmitter:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))


def _environment() -> dict[str, str]:
    # Allowlisted hosts only — no custom-endpoint opt-in needed.
    return {
        "OPENROUTER_API_KEY": "or-test",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENROUTER_MODEL": "openrouter-model",
        "OPENCODE_API_KEY": "oc-test",
        "OPENCODE_API_BASE_URL": "https://opencode.ai/zen/go/v1",
        "OPENCODE_MODEL": "opencode-model",
    }


def _request_bodies(mocked) -> list[bytes]:
    return [call.args[0].data for call in mocked.call_args_list]


def _request_hosts(mocked) -> list[str]:
    return [call.args[0].host for call in mocked.call_args_list]


def _request_timeouts(mocked) -> list[float]:
    return [call.kwargs["timeout"] for call in mocked.call_args_list]


def _prompt_of(body: bytes) -> str:
    return json.loads(body.decode("utf-8"))["messages"][0]["content"]


class MeetingMinutesTests(unittest.TestCase):
    def test_openrouter_failure_falls_back_to_opencode(self) -> None:
        # 503 is retryable: two attempts on OpenRouter, then OpenCode succeeds.
        with patch("mediascribe_cli.meeting_minutes.sleep"), patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[_http_error(503), _http_error(503), FakeResponse()],
        ) as mocked:
            generated = generate_meeting_minutes(
                "[00:00:01] A: 일정 논의", "제품 회의", environment=_environment()
            )

        self.assertEqual(generated.provider, "opencode")
        self.assertEqual(generated.model, "opencode-model")
        self.assertIn("일정 논의", generated.text)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(_request_hosts(mocked), ["openrouter.ai", "openrouter.ai", "opencode.ai"])

    def test_non_retryable_error_falls_back_immediately(self) -> None:
        # 401 is non-retryable: no second attempt on OpenRouter.
        with patch("mediascribe_cli.meeting_minutes.sleep") as slept, patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[_http_error(401, b'{"error":{"message":"invalid key"}}'), FakeResponse()],
        ) as mocked:
            generated = generate_body_summary("전사", environment=_environment())

        self.assertEqual(generated.provider, "opencode")
        self.assertEqual(mocked.call_count, 2)
        slept.assert_not_called()

    def test_redirect_response_is_rejected_without_follow(self) -> None:
        # A 302 must surface as ProviderError — the no-redirect opener never
        # forwards Authorization to another host. Only OpenRouter is keyed,
        # so a single request is made and no second call follows.
        environment = _environment()
        environment.pop("OPENCODE_API_KEY")
        with patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[_http_error(302)],
        ) as mocked:
            with self.assertRaises(ProviderError):
                generate_body_summary("전사", environment=environment)
        self.assertEqual(mocked.call_count, 1)

    def test_no_redirect_handler_refuses_redirect_request(self) -> None:
        handler = _NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/")
        )

    def test_retryable_error_retries_then_uses_checkpoint(self) -> None:
        transcript = "\n".join(f"[00:{i:02d}:00] A: 발화 {i}" for i in range(60))
        checkpoint = DictCheckpoint()
        with patch("mediascribe_cli.meeting_minutes.sleep") as slept, patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[URLError("boom"), FakeResponse("부분 요약 1"), FakeResponse("부분 요약 2"), FakeResponse("최종 요약")],
        ) as mocked:
            generated = generate_body_summary(
                transcript, environment=_environment(), max_chunk_chars=1024, checkpoint=checkpoint
            )

        self.assertEqual(generated.provider, "openrouter")
        chunks = chunk_transcript(transcript.splitlines(), 1024)
        self.assertEqual(len(chunks), 2)
        # chunk1: fail + retry, chunk2: first try, reduce: first try → 4 calls.
        self.assertEqual(mocked.call_count, 4)
        slept.assert_called_once_with(1.0)
        # Keys are namespaced digests: "<ns>:sha256:<chunk-digest>".
        self.assertEqual(len(checkpoint.put_calls), 2)
        for key, chunk in zip(checkpoint.put_calls, chunks):
            self.assertTrue(key.endswith(":" + chunk_digest(chunk)))

    def test_chunked_map_reduce_calls_and_bounded_bodies(self) -> None:
        transcript = "\n".join(f"[00:{i:02d}:00] A: 발화 {i} 내용" for i in range(90))
        chunks = chunk_transcript(transcript.splitlines(), 1024)
        self.assertEqual(len(chunks), 3)
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")) as mocked:
            generated = generate_body_summary(
                transcript, environment=_environment(), max_chunk_chars=1024
            )

        # 3 map calls + 1 reduce call, order preserved.
        self.assertEqual(mocked.call_count, 4)
        bodies = _request_bodies(mocked)
        for body in bodies:
            self.assertLessEqual(len(body), MAX_REQUEST_BODY_BYTES)
        prompts = [_prompt_of(body) for body in bodies]
        for index, prompt in enumerate(prompts[:3]):
            self.assertIn(f"전체 3개 구간 중 {index + 1}번째", prompt)
        self.assertIn("구간별 부분 요약", prompts[3])
        self.assertEqual(generated.provider, "openrouter")

    def test_chunk_count_cap_rejects_oversized_transcript(self) -> None:
        # A transcript that would need more than MAX_SUMMARY_CHUNKS provider
        # calls is rejected up front — no unbounded request amplification.
        from mediascribe_cli.meeting_minutes import MAX_SUMMARY_CHUNKS
        transcript = "\n".join(f"[00:{i:02d}:00] A: " + "x" * 100 for i in range(MAX_SUMMARY_CHUNKS * 12))
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")) as mocked:
            with self.assertRaises(ProviderError) as caught:
                generate_body_summary(transcript, environment=_environment(), max_chunk_chars=1024)
        self.assertIn("구간", str(caught.exception))
        mocked.assert_not_called()


    def test_summary_fingerprint_tracks_effective_provider(self) -> None:
        # Auto mode: the fingerprint must change when the *effective*
        # provider set changes — same model name served by a different
        # provider is a different summary source.
        from mediascribe_cli.meeting_minutes import summary_config_fingerprint

        env_router = {"OPENROUTER_API_KEY": "k1", "OPENROUTER_MODEL": "same-model"}
        env_code = {"OPENCODE_API_KEY": "k2", "OPENCODE_MODEL": "same-model"}
        fp_router = summary_config_fingerprint(env_router, "", "제목", 24000)
        fp_code = summary_config_fingerprint(env_code, "", "제목", 24000)
        self.assertNotEqual(fp_router, fp_code)
        # Same env → identical fingerprint (deterministic).
        self.assertEqual(fp_router, summary_config_fingerprint(dict(env_router), "", "제목", 24000))
    def test_checkpoint_hit_skips_chunk_request(self) -> None:
        transcript = "\n".join(f"[00:{i:02d}:00] A: 발화 {i}" for i in range(60))
        chunks = chunk_transcript(transcript.splitlines(), 1024)
        self.assertEqual(len(chunks), 2)
        # Discover the namespaced key for chunk 1 from a first pass.
        seeder = DictCheckpoint()
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")):
            generate_body_summary(
                transcript, environment=_environment(), max_chunk_chars=1024, checkpoint=seeder
            )
        first_key = seeder.put_calls[0]
        self.assertTrue(first_key.endswith(":" + chunk_digest(chunks[0])))

        checkpoint = DictCheckpoint({first_key: "저장된 부분 요약"})
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")) as mocked:
            generate_body_summary(
                transcript, environment=_environment(), max_chunk_chars=1024, checkpoint=checkpoint
            )

        # chunk1 served from checkpoint → only chunk2 map + reduce.
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(checkpoint.put_calls), 1)
        self.assertTrue(checkpoint.put_calls[0].endswith(":" + chunk_digest(chunks[1])))
        self.assertEqual(checkpoint.get_calls[0], first_key)

    def test_checkpoint_namespace_covers_prompt_config(self) -> None:
        # Different title or provider set → different namespace → no stale reuse.
        transcript = "\n".join(f"[00:{i:02d}:00] A: 발화 {i}" for i in range(60))

        def keys_for(**kwargs):
            store = DictCheckpoint()
            with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")):
                generate_body_summary(
                    transcript,
                    environment=_environment(),
                    max_chunk_chars=1024,
                    checkpoint=store,
                    **kwargs,
                )
            return store.put_calls

        base = keys_for(title="회의 A")
        other_title = keys_for(title="회의 B")
        other_provider = keys_for(title="회의 A", provider="opencode")
        self.assertNotEqual(base, other_title)
        self.assertNotEqual(base, other_provider)
        # Same config → identical namespaced keys (deterministic resume).
        self.assertEqual(base, keys_for(title="회의 A"))

    def test_single_chunk_uses_summary_prompt_without_reduce(self) -> None:
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse()) as mocked:
            generated = generate_body_summary("짧은 전사", environment=_environment())

        self.assertEqual(mocked.call_count, 1)
        prompt = _prompt_of(_request_bodies(mocked)[0])
        self.assertIn("본문 요약", prompt)
        self.assertNotIn("구간", prompt)
        self.assertEqual(generated.provider, "openrouter")

    def test_oversized_response_is_non_retryable(self) -> None:
        oversized = FakeResponse(raw=b"x" * (MAX_RESPONSE_BYTES + 1))
        with patch("mediascribe_cli.meeting_minutes.sleep") as slept, patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[oversized, FakeResponse()],
        ) as mocked:
            generated = generate_body_summary("전사", environment=_environment())

        self.assertEqual(mocked.call_count, 2)  # no retry on openrouter
        slept.assert_not_called()
        self.assertEqual(generated.provider, "opencode")

    def test_error_body_is_truncated(self) -> None:
        huge = b"e" * (MAX_ERROR_BODY_BYTES + 4096)
        with patch("mediascribe_cli.meeting_minutes.sleep"), patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[_http_error(500, huge), _http_error(500, huge), _http_error(500, huge), _http_error(500, huge)],
        ):
            with self.assertRaises(ProviderError) as ctx:
                generate_body_summary("전사", environment=_environment())
        # Each provider's error body is truncated to MAX_ERROR_BODY_BYTES; the
        # aggregate failure joins at most two bounded provider messages.
        self.assertLessEqual(len(str(ctx.exception)), 2 * (MAX_ERROR_BODY_BYTES + 512))

    def test_api_key_never_appears_in_errors(self) -> None:
        environment = _environment()
        environment["OPENROUTER_API_KEY"] = "or-secret-key-12345"
        environment["OPENCODE_API_KEY"] = "oc-secret-key-67890"
        leaky_body = b'{"error":{"message":"rejected or-secret-key-12345"}}'
        with patch("mediascribe_cli.meeting_minutes.sleep"), patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[
                _http_error(401, leaky_body),
                URLError("refused by oc-secret-key-67890"),
                URLError("refused by oc-secret-key-67890"),
            ],
        ):
            with self.assertRaises(ProviderError) as ctx:
                generate_body_summary("전사", environment=environment)
        message = str(ctx.exception)
        self.assertNotIn("or-secret-key-12345", message)
        self.assertNotIn("oc-secret-key-67890", message)

    def test_timeout_is_clamped(self) -> None:
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse()) as mocked:
            generate_body_summary("전사", environment=_environment(), timeout=0.001)
        self.assertEqual(_request_timeouts(mocked), [5.0])
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse()) as mocked:
            generate_body_summary("전사", environment=_environment(), timeout=99999)
        self.assertEqual(_request_timeouts(mocked), [600.0])

    def test_progress_events_have_real_elapsed(self) -> None:
        transcript = "\n".join(f"[00:{i:02d}:00] A: 발화 {i}" for i in range(60))
        emitter = RecordingEmitter()
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")):
            generate_body_summary(
                transcript, environment=_environment(), max_chunk_chars=1024, emitter=emitter
            )

        progress = [payload for kind, payload in emitter.events if kind == "progress"]
        self.assertTrue(progress)
        self.assertEqual(progress[-1]["progress"], 100.0)
        self.assertGreater(progress[-1]["elapsed_seconds"], 0)
        percentages = [p["progress"] for p in progress]
        self.assertEqual(percentages[0], 0.0)
        self.assertIn(50.0, percentages)

    def test_provider_failed_event_emitted_per_exhausted_provider(self) -> None:
        emitter = RecordingEmitter()
        with patch("mediascribe_cli.meeting_minutes.sleep"), patch(
            "mediascribe_cli.meeting_minutes.urlopen",
            side_effect=[_http_error(401), FakeResponse()],
        ):
            generate_body_summary("전사", environment=_environment(), emitter=emitter)
        failed = [p for kind, p in emitter.events if kind == "provider_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["provider"], "openrouter")
        self.assertIn("401", failed[0]["error"])

    def test_custom_endpoint_requires_opt_in(self) -> None:
        environment = _environment()
        environment["OPENROUTER_BASE_URL"] = "https://openrouter.example.com/v1"
        with self.assertRaises(ConfigError):
            generate_body_summary("전사", environment=environment)
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse()) as mocked:
            generated = generate_body_summary(
                "전사", environment=environment, allow_custom_endpoint=True
            )
        self.assertEqual(generated.provider, "openrouter")
        self.assertEqual(_request_hosts(mocked), ["openrouter.example.com"])

    def test_meeting_minutes_uses_same_bounded_chunked_path(self) -> None:
        transcript = "\n".join(f"[00:{i:02d}:00] A: 발화 {i}" for i in range(60))
        with patch("mediascribe_cli.meeting_minutes.urlopen", return_value=FakeResponse("요약")) as mocked:
            generated = generate_meeting_minutes(
                transcript, environment=_environment(), max_chunk_chars=1024
            )
        self.assertEqual(mocked.call_count, 3)  # 2 map + 1 reduce
        self.assertEqual(generated.provider, "openrouter")

    def test_minutes_command_writes_provider_result(self) -> None:
        from mediascribe_cli.cli import main  # lazy: cli.py may be mid-migration

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "meeting_speaker.txt"
            destination = Path(directory) / "meeting_본문요약.md"
            source.write_text("[00:00:01–00:00:03] A: 일정 논의", encoding="utf-8")
            generated = GeneratedMinutes("# 제품 회의\n\n## 핵심 요약\n- 일정 논의\n", "openrouter", "test-model")
            with patch("mediascribe_cli.cli.generate_body_summary", return_value=generated):
                exit_code = main(["minutes", str(source), "--output", str(destination), "--title", "제품 회의"])
            self.assertEqual(exit_code, 0)
            self.assertIn("일정 논의", destination.read_text(encoding="utf-8"))

    def test_missing_provider_keys_is_explicit(self) -> None:
        with self.assertRaisesRegex(ProviderError, "OPENROUTER_API_KEY"):
            generate_meeting_minutes("전사", environment={})


if __name__ == "__main__":
    unittest.main()
