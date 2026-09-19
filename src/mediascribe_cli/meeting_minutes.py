from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from .chunking import (
    DEFAULT_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    MAX_ERROR_BODY_BYTES,
    MAX_RESPONSE_BYTES,
    chunk_digest,
    chunk_transcript,
)
from .config import _candidate_specs, load_environment, provider_candidates
from .errors import ProviderError
from .events import EventEmitter

# Request/response bounds (production-readiness §4.5).
MIN_CHUNK_CHARS = 1_024
PROMPT_OVERHEAD_BYTES = 8 * 1_024
# Chunks are sized in characters but the cap is enforced on UTF-8 bytes;
# 4 bytes/char is the worst case, so Korean text (~3 bytes/char) still fits.
MAX_REQUEST_BODY_BYTES = 4 * MAX_CHUNK_CHARS + PROMPT_OVERHEAD_BYTES
MIN_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 600.0
MAX_ATTEMPTS_PER_PROVIDER = 2
RETRYABLE_HTTP_CODES = {408, 409, 425, 429}
MAX_REDUCE_ROUNDS = 2
MAX_SUMMARY_CHUNKS = 512  # hard cap on provider calls per summary; raise --summary-chunk-chars for longer transcripts
TRUNCATION_MARKER = "\n\n[…후략 — 전사가 예산을 초과했습니다]"


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse every redirect: a 30x must surface as HTTPError, never forward
    the Authorization header to a host outside the endpoint allowlist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


# Module-level name kept so tests can patch ``meeting_minutes.urlopen``.
urlopen = build_opener(_NoRedirectHandler()).open



class SummaryCheckpoint(Protocol):
    """Digest-keyed store for per-chunk partial summaries."""

    def get(self, digest: str) -> str | None: ...

    def put(self, digest: str, text: str) -> None: ...


class _RetryableProviderError(ProviderError):
    """Provider failure worth retrying on the same provider (network, 408/409/425/429/5xx)."""


@dataclass(frozen=True)
class GeneratedMinutes:
    text: str
    provider: str
    model: str


def build_prompt(title: str, transcript: str) -> str:
    return f"""당신은 근거 중심의 한국어 회의록 편집자입니다.
아래 화자 전사본만 사용해 실무용 회의록을 작성하세요.

규칙:
- 전사본에 없는 사실, 숫자, 담당자, 기한, 결론을 만들지 마세요.
- 화자 이름이 없으면 A/B/C 같은 화자 표기를 그대로 사용하세요.
- 결정사항, 제안, 질문, 미확정 사항을 구분하세요.
- 가격·수량·기간은 발언상 수치인지 합의된 조건인지 구분하세요.
- 각 핵심 항목과 액션에는 가능한 한 [HH:MM:SS–HH:MM:SS] 근거 시간을 붙이세요.
- 확실하지 않은 회사명·제품명·기술명은 ‘확인 필요’로 표시하세요.
- 결과는 마크다운으로만 반환하세요.

권장 형식:
# {title}
## 핵심 요약
## 주요 논의
## 결정사항
## 후속 액션
| 액션 | 담당자 | 기한 | 근거 |
## 미확정·리스크
## 확인 필요 용어

화자 전사본:
{transcript}"""


def build_summary_prompt(title: str, transcript: str) -> str:
    return f"""당신은 근거 중심의 한국어 본문 요약 편집자입니다.
아래 화자 전사본만 사용해 짧고 정확한 본문 요약을 작성하세요.

규칙:
- 전사본에 없는 사실, 숫자, 담당자, 기한, 결론을 만들지 마세요.
- 핵심 주제와 발언 흐름을 압축하되 중요한 조건은 생략하지 마세요.
- 결정사항, 후속 액션, 미확정 사항을 각각 구분하세요.
- 각 항목에는 가능한 한 [HH:MM:SS–HH:MM:SS] 근거 시간을 붙이세요.
- 확실하지 않은 용어는 ‘확인 필요’로 표시하세요.
- 결과는 마크다운으로만 반환하세요.

권장 형식:
# {title}
## 핵심 요약
## 주요 내용
## 결정사항
## 후속 액션
## 미확정·확인 필요

화자 전사본:
{transcript}"""


def build_chunk_prompt(title: str, index: int, total: int, chunk_text: str) -> str:
    return f"""당신은 근거 중심의 한국어 본문 요약 편집자입니다.
아래는 회의 「{title}」의 화자 전사본 일부입니다. 이것은 전체 {total}개 구간 중 {index}번째입니다.
이 구간에 나타난 내용만 사용해 부분 요약을 작성하세요.

규칙:
- 전사본에 없는 사실, 숫자, 담당자, 기한, 결론을 만들지 마세요.
- 핵심 주제와 발언 흐름을 압축하되 중요한 조건은 생략하지 마세요.
- 결정사항, 후속 액션, 미확정 사항을 각각 구분하세요.
- 각 항목에는 가능한 한 [HH:MM:SS–HH:MM:SS] 근거 시간을 붙이세요.
- 확실하지 않은 용어는 ‘확인 필요’로 표시하세요.
- 이 구간 범위를 벗어나는 내용은 추측하지 마세요.
- 결과는 마크다운으로만 반환하세요.

화자 전사본 ({index}/{total} 구간):
{chunk_text}"""


def build_reduce_prompt(title: str, partials_text: str) -> str:
    return f"""당신은 근거 중심의 한국어 본문 요약 편집자입니다.
아래는 회의 「{title}」의 구간별 부분 요약입니다. 중복을 제거하고 하나의 본문 요약으로 통합하세요.

규칙:
- 부분 요약에 없는 사실, 숫자, 담당자, 기한, 결론을 만들지 마세요.
- 핵심 주제와 발언 흐름을 압축하되 중요한 조건은 생략하지 마세요.
- 결정사항, 후속 액션, 미확정 사항을 각각 구분하세요.
- 각 항목에는 가능한 한 [HH:MM:SS–HH:MM:SS] 근거 시간을 유지하세요.
- 확실하지 않은 용어는 ‘확인 필요’로 표시하세요.
- 결과는 마크다운으로만 반환하세요.

권장 형식:
# {title}
## 핵심 요약
## 주요 내용
## 결정사항
## 후속 액션
## 미확정·확인 필요

구간별 부분 요약:
{partials_text}"""


def _extract_content(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    return str(content or "").strip()


def _headers(provider_id: str, api_key: str, environment: Mapping[str, str]) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider_id == "openrouter":
        title = environment.get("OPENROUTER_APP_TITLE", "MediaScribe-CLI").strip()
        referer = environment.get("OPENROUTER_HTTP_REFERER", "").strip()
        if title:
            headers["X-Title"] = title
        if referer:
            headers["HTTP-Referer"] = referer
    return headers


def _redact_key(text: Any, api_key: str) -> str:
    """Guarantee the provider API key never reaches an error message or log."""
    value = str(text)
    return value.replace(api_key, "***") if api_key else value


def _clamp_timeout(timeout: float) -> float:
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        value = 120.0
    return min(max(value, MIN_TIMEOUT_SECONDS), MAX_TIMEOUT_SECONDS)


def _clamp_chunk_chars(max_chunk_chars: int) -> int:
    try:
        value = int(max_chunk_chars)
    except (TypeError, ValueError):
        value = DEFAULT_CHUNK_CHARS
    if value <= 0:
        value = DEFAULT_CHUNK_CHARS
    return min(max(value, MIN_CHUNK_CHARS), MAX_CHUNK_CHARS)


def _error_detail(exc: HTTPError, api_key: str) -> str:
    try:
        raw = exc.read(MAX_ERROR_BODY_BYTES + 1)
    except Exception:
        raw = b""
    if isinstance(raw, (bytes, bytearray)):
        detail = bytes(raw[:MAX_ERROR_BODY_BYTES]).decode("utf-8", errors="replace")
    else:
        detail = str(raw)[:MAX_ERROR_BODY_BYTES]
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, Mapping):
            error = parsed.get("error")
            detail = str(
                (error.get("message") if isinstance(error, Mapping) else None)
                or parsed.get("message")
                or detail
            )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError, MemoryError):
        pass
    return _redact_key(detail, api_key)


def _request(provider, prompt: str, environment: Mapping[str, str], timeout: float) -> str:
    endpoint = f"{provider.base_url.rstrip('/')}/chat/completions"
    body = json.dumps(
        {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 8192,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise ProviderError(
            f"{provider.label} 요청 본문이 허용 한도({MAX_REQUEST_BODY_BYTES}바이트)를 초과합니다."
        )
    request = Request(endpoint, data=body, headers=_headers(provider.id, provider.api_key, environment), method="POST")
    bounded_timeout = _clamp_timeout(timeout)
    try:
        with urlopen(request, timeout=bounded_timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except HTTPError as exc:
        detail = _error_detail(exc, provider.api_key)
        error_type = (
            _RetryableProviderError
            if exc.code in RETRYABLE_HTTP_CODES or exc.code >= 500
            else ProviderError
        )
        raise error_type(f"{provider.label} API 오류 ({exc.code}): {detail or '응답 없음'}") from exc
    except URLError as exc:
        raise _RetryableProviderError(
            f"{provider.label} 네트워크 요청 실패: {_redact_key(exc.reason, provider.api_key)}"
        ) from exc
    except TimeoutError as exc:
        raise _RetryableProviderError(
            f"{provider.label} 요청이 {bounded_timeout:.0f}초 후 시간 초과되었습니다."
        ) from exc
    except OSError as exc:
        raise _RetryableProviderError(
            f"{provider.label} 요청이 실패했습니다: {_redact_key(exc, provider.api_key)}"
        ) from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError(
            f"{provider.label} 응답이 허용 한도({MAX_RESPONSE_BYTES}바이트)를 초과합니다."
        )
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise ProviderError(f"{provider.label}가 JSON이 아닌 응답을 반환했습니다 (HTTP {status}).") from exc
    text = _extract_content(payload)
    if not text:
        raise ProviderError(f"{provider.label}가 빈 회의록을 반환했습니다.")
    return text


def _request_with_retry(
    candidates,
    prompt: str,
    environment: Mapping[str, str],
    timeout: float,
    emitter: EventEmitter | None,
) -> tuple[str, Any]:
    """Try each provider in order; ≤2 attempts each, ≤4 HTTP calls total.

    Retryable failures (network, 408/409/425/429/5xx) retry on the same
    provider after a short fixed delay; non-retryable failures fall back to
    the next provider immediately.
    """
    failures: list[str] = []
    for candidate in candidates:
        last_error: ProviderError | None = None
        for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
            try:
                return _request(candidate, prompt, environment, timeout), candidate
            except _RetryableProviderError as exc:
                last_error = exc
                sleep(1.0 * attempt)  # 1s, 2s — fixed, no jitter
            except ProviderError as exc:
                last_error = exc
                break  # non-retryable → next provider immediately
        failures.append(str(last_error))
        if emitter is not None:
            emitter.emit("provider_failed", provider=candidate.id, error=str(last_error))
    raise ProviderError(f"모든 공급자 요청에 실패했습니다. {' | '.join(failures)}")


def _reduce(title: str, partials: list[str], budget: int, call) -> tuple[str, Any]:
    """Combine partial summaries within the request budget.

    One reduce call when the joined partials fit; otherwise group partials by
    the chunk budget and reduce each group, for at most MAX_REDUCE_ROUNDS
    rounds. Still over budget → deterministic truncation + one final reduce.
    Bounded and always terminates.
    """
    current = [str(partial) for partial in partials]
    for _round in range(MAX_REDUCE_ROUNDS):
        joined = "\n\n".join(current)
        if len(joined) <= budget:
            return call(build_reduce_prompt(title, joined))
        groups = chunk_transcript(current, budget)
        if len(groups) > len(current):
            break  # hard-split pieces can only grow the input → deterministic truncation
        current = [call(build_reduce_prompt(title, "\n\n".join(group)))[0] for group in groups]
    joined = "\n\n".join(current)
    truncated = joined[:budget] + TRUNCATION_MARKER
    return call(build_reduce_prompt(title, truncated))


def _generate_chunked(
    transcript: str,
    title: str,
    environment: Mapping[str, str] | None,
    provider: str,
    timeout: float,
    leaf_prompt_builder,
    *,
    max_chunk_chars: int,
    emitter: EventEmitter | None,
    allow_custom_endpoint: bool,
    checkpoint: SummaryCheckpoint | None,
) -> GeneratedMinutes:
    normalized = str(transcript or "").strip()
    if not normalized:
        raise ProviderError("요약을 생성할 전사 텍스트가 없습니다.")
    env = dict(load_environment() if environment is None else environment)
    candidates = provider_candidates(
        env,
        provider=provider,
        allow_custom_endpoint=allow_custom_endpoint,
        emitter=emitter,
    )
    if not candidates:
        raise ProviderError("OPENROUTER_API_KEY 또는 OPENCODE_API_KEY가 설정되지 않았습니다.")

    resolved_title = title or "회의록"
    budget = _clamp_chunk_chars(max_chunk_chars)
    # Checkpoint keys are namespaced by everything that shapes the prompt:
    # provider + candidate models + title + prompt builder + format version.
    # Resuming with a different title/provider/model must not reuse stale
    # partial summaries.
    namespace_material = "|".join(
        [
            str(provider or ""),
            ",".join(str(candidate.model) for candidate in candidates),
            resolved_title,
            getattr(leaf_prompt_builder, "__name__", repr(leaf_prompt_builder)),
            "v1",
        ]
    )
    namespace = hashlib.sha256(namespace_material.encode("utf-8")).hexdigest()[:16]

    started_at = perf_counter()

    def call(prompt: str) -> tuple[str, Any]:
        return _request_with_retry(candidates, prompt, env, timeout, emitter)

    def finish(text: str, candidate) -> GeneratedMinutes:
        if not text.startswith("#"):
            text = f"# {resolved_title}\n\n{text}\n"
        return GeneratedMinutes(text=text, provider=candidate.id, model=candidate.model)

    chunks = chunk_transcript(normalized.splitlines(), budget)
    if len(chunks) > MAX_SUMMARY_CHUNKS:
        raise ProviderError(
            f"전사가 {len(chunks)}개 구간으로 나뉘어 요약 한도({MAX_SUMMARY_CHUNKS}개)를 초과합니다. "
            f"--summary-chunk-chars를 늘리세요."
        )
    if len(chunks) <= 1:
        text, candidate = call(leaf_prompt_builder(resolved_title, normalized))
        if emitter is not None:
            emitter.emit(
                "progress",
                stage="summary",
                progress=100.0,
                elapsed_seconds=perf_counter() - started_at,
                text="본문 요약 완료",
            )
        return finish(text, candidate)

    total = len(chunks)
    partials: list[str] = []
    for index, chunk in enumerate(chunks):
        digest = f"{namespace}:{chunk_digest(chunk)}"
        text = checkpoint.get(digest) if checkpoint is not None else None
        if text is None:
            if emitter is not None:
                emitter.emit(
                    "progress",
                    stage="summary",
                    progress=index / total * 100.0,
                    elapsed_seconds=perf_counter() - started_at,
                    text=f"구간 {index + 1}/{total} 요약 중",
                )
            text, _candidate = call(
                build_chunk_prompt(resolved_title, index + 1, total, "\n".join(chunk))
            )
            if checkpoint is not None:
                checkpoint.put(digest, text)
        partials.append(text)

    text, candidate = _reduce(resolved_title, partials, budget, call)
    if emitter is not None:
        emitter.emit(
            "progress",
            stage="summary",
            progress=100.0,
            elapsed_seconds=perf_counter() - started_at,
            text="본문 요약 완료",
        )
    return finish(text, candidate)


def generate_meeting_minutes(
    transcript: str,
    title: str = "회의록",
    environment: Mapping[str, str] | None = None,
    provider: str = "",
    timeout: float = 120.0,
    *,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
    emitter: EventEmitter | None = None,
    allow_custom_endpoint: bool = False,
    checkpoint: SummaryCheckpoint | None = None,
) -> GeneratedMinutes:
    return _generate_chunked(
        transcript,
        title,
        environment,
        provider,
        timeout,
        build_prompt,
        max_chunk_chars=max_chunk_chars,
        emitter=emitter,
        allow_custom_endpoint=allow_custom_endpoint,
        checkpoint=checkpoint,
    )


def generate_body_summary(
    transcript: str,
    title: str = "회의록",
    environment: Mapping[str, str] | None = None,
    provider: str = "",
    timeout: float = 120.0,
    *,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
    emitter: EventEmitter | None = None,
    allow_custom_endpoint: bool = False,
    checkpoint: SummaryCheckpoint | None = None,
) -> GeneratedMinutes:
    return _generate_chunked(
        transcript,
        title,
        environment,
        provider,
        timeout,
        build_summary_prompt,
        max_chunk_chars=max_chunk_chars,
        emitter=emitter,
        allow_custom_endpoint=allow_custom_endpoint,
        checkpoint=checkpoint,
    )


def summary_config_fingerprint(
    environment: Mapping[str, str] | None,
    provider: str,
    title: str,
    max_chunk_chars: int,
) -> str:
    """Fingerprint of the inputs that shape a summary *other than* the
    transcript itself: the *effective* provider candidate set (which keys
    are present and which endpoint/model each resolves to — never the key
    material), title, and chunk budget. Stored in the manifest at commit
    time; ``should_skip`` requires it to match so a rerun with a different
    title/provider/model — or a different provider serving the same model
    name — re-summarizes instead of silently serving a stale artifact."""
    env = dict(load_environment() if environment is None else environment)
    candidates = _candidate_specs(env, provider)
    material = "|".join(
        [
            str(provider or ""),
            *(f"{spec['id']}:{spec['base_url']}:{spec['model']}" for spec in candidates),
            str(title or ""),
            str(int(max_chunk_chars)),
            "v2",
        ]
    )
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
