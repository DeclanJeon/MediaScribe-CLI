from __future__ import annotations

import ipaddress
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from .artifacts import DEFAULT_ARTIFACT_FORMAT
from .chunking import DEFAULT_CHUNK_CHARS
from .errors import ConfigError


DEFAULT_MODEL = "small"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731:free"
DEFAULT_OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_OPENCODE_MODEL = "deepseek-v4-flash"

# Only these hosts may ever receive API keys or transcript content.
ALLOWED_PROVIDER_HOSTS = {
    "openrouter": {"openrouter.ai"},
    "opencode": {"opencode.ai"},
}

_SECRET_KEY_MARKERS = ("_KEY", "_TOKEN")


def _parse_env_file(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value.replace("\\n", "\n").replace("\\r", "\r")
    return values


def _user_env_path() -> Path:
    return Path.home() / ".config" / "mediascribe" / "env"


def _warn_secret_file_permissions(path: Path, values: Mapping[str, str]) -> None:
    """Warn on stderr when a loaded env file holds secrets but is group/world-readable."""
    if not any(key.endswith(_SECRET_KEY_MARKERS) for key in values):
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & 0o077:
        print(
            f"경고: {path} 에 비밀 값이 있지만 그룹/다른 사용자가 읽을 수 있습니다. "
            "`chmod 600`으로 권한을 제한하세요.",
            file=sys.stderr,
        )


_MAX_ENV_FILE_BYTES = 1024 * 1024


def _load_env_file(path: Path, *, explicit: bool) -> dict[str, str]:
    try:
        if path.stat().st_size > _MAX_ENV_FILE_BYTES:
            if explicit:
                raise ConfigError(f"환경 파일이 허용 한도(1 MiB)를 초과합니다: {path}")
            return {}
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        if explicit:
            raise ConfigError(f"환경 파일을 읽을 수 없습니다: {path}") from exc
        return {}
    values = _parse_env_file(content)
    _warn_secret_file_permissions(path, values)
    return values


def load_environment(env_file: str | None = None) -> dict[str, str]:
    """Load environment values. First source wins:

    1. process environment
    2. explicit env file (--env-file argument or MEDIASCRIBE_ENV_FILE)
    3. ~/.config/mediascribe/env (user-level, never cwd/package files)

    The working-directory and package .env* files are intentionally NOT read:
    a hostile cwd could redirect API keys and transcripts elsewhere.
    """
    values = dict(os.environ)
    configured = str(env_file if env_file is not None else values.get("MEDIASCRIBE_ENV_FILE", "")).strip()
    if configured:
        explicit_path = Path(configured).expanduser()
        if not explicit_path.is_file():
            raise ConfigError(f"환경 파일이 없습니다: {explicit_path}")
        for key, value in _load_env_file(explicit_path, explicit=True).items():
            values.setdefault(key, value)
    user_path = _user_env_path()
    if user_path.is_file():
        for key, value in _load_env_file(user_path, explicit=False).items():
            values.setdefault(key, value)
    return values


def _int_value(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return default


def _bool_value(value: str | None, default: bool = False) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RuntimeConfig:
    model: str = DEFAULT_MODEL
    language: str = ""
    task: str = "transcribe"
    device: str = "auto"
    compute_type: str = ""
    beam_size: int = 1
    vad_filter: bool = True
    diarize: bool = False
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    num_speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0
    glossary_path: str = ""
    speaker_map_path: str = ""
    speaker_labels: tuple[str, ...] = ()
    artifact_format: str = DEFAULT_ARTIFACT_FORMAT
    json_lines: bool = False
    env_file: str = ""
    allow_custom_endpoint: bool = False
    resume: bool = True
    follow_symlinks: bool = False
    summary_chunk_chars: int = DEFAULT_CHUNK_CHARS
    events_full_text: bool = False


def build_runtime_config(args, environment: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = dict(load_environment() if environment is None else environment)
    artifact_format = str(getattr(args, "output_format", None) or env.get("MEDIASCRIBE_OUTPUT_FORMAT", DEFAULT_ARTIFACT_FORMAT)).strip().lower()
    raw_labels = getattr(args, "speaker_labels", None) or env.get("MEDIASCRIBE_SPEAKER_LABELS", "")
    labels = tuple(token.strip() for token in str(raw_labels).split(",") if token.strip())
    arg_env_file = str(getattr(args, "env_file", None) or "").strip()
    arg_chunk_chars = getattr(args, "summary_chunk_chars", None)
    arg_no_resume = getattr(args, "no_resume", None)
    return RuntimeConfig(
        model=getattr(args, "model", None) or env.get("MEDIASCRIBE_MODEL", DEFAULT_MODEL),
        language=getattr(args, "language", None) or env.get("MEDIASCRIBE_LANGUAGE", ""),
        task=getattr(args, "task", None) or env.get("MEDIASCRIBE_TASK", "transcribe"),
        device=getattr(args, "device", None) or env.get("MEDIASCRIBE_DEVICE", "auto"),
        compute_type=getattr(args, "compute_type", None) or env.get("MEDIASCRIBE_COMPUTE_TYPE", ""),
        beam_size=getattr(args, "beam_size", None) or _int_value(env.get("MEDIASCRIBE_BEAM_SIZE"), 1),
        vad_filter=not bool(getattr(args, "no_vad_filter", False)),
        diarize=bool(getattr(args, "diarize", False)),
        diarization_model=getattr(args, "diarization_model", None) or env.get("MEDIASCRIBE_DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL),
        num_speakers=getattr(args, "num_speakers", None) or _int_value(env.get("MEDIASCRIBE_NUM_SPEAKERS"), 0),
        min_speakers=getattr(args, "min_speakers", None) or _int_value(env.get("MEDIASCRIBE_MIN_SPEAKERS"), 0),
        max_speakers=getattr(args, "max_speakers", None) or _int_value(env.get("MEDIASCRIBE_MAX_SPEAKERS"), 0),
        glossary_path=getattr(args, "glossary", None) or env.get("MEDIASCRIBE_GLOSSARY", ""),
        speaker_map_path=getattr(args, "speaker_map", None) or env.get("MEDIASCRIBE_SPEAKER_MAP", ""),
        speaker_labels=labels,
        artifact_format=artifact_format,
        json_lines=bool(getattr(args, "json_lines", False)),
        env_file=arg_env_file or env.get("MEDIASCRIBE_ENV_FILE", "").strip(),
        allow_custom_endpoint=bool(getattr(args, "allow_custom_endpoint", False))
        or _bool_value(env.get("MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT"), False),
        resume=(not arg_no_resume) if arg_no_resume is not None else _bool_value(env.get("MEDIASCRIBE_RESUME"), True),
        follow_symlinks=bool(getattr(args, "follow_symlinks", False))
        or _bool_value(env.get("MEDIASCRIBE_FOLLOW_SYMLINKS"), False),
        summary_chunk_chars=_int_value(arg_chunk_chars, 0)
        if arg_chunk_chars
        else _int_value(env.get("MEDIASCRIBE_SUMMARY_CHUNK_CHARS"), DEFAULT_CHUNK_CHARS),
        events_full_text=bool(getattr(args, "events_full_text", False))
        or _bool_value(env.get("MEDIASCRIBE_EVENTS_FULL_TEXT"), False),
    )


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    label: str
    api_key: str
    base_url: str
    model: str


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _safe_url_display(raw_url: str, parts) -> str:
    host = parts.hostname or "?"
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        port = ""
    userinfo = "***@" if parts.username or parts.password else ""
    return f"{parts.scheme or '(no scheme)'}://{userinfo}{host}{port}"


def _validate_base_url(provider_id: str, raw_url: str, *, allow_custom_endpoint: bool, emitter) -> str:
    url = str(raw_url or "").strip()
    if not url:
        raise ConfigError(f"{provider_id} base URL이 비어 있습니다.")
    try:
        parts = urlsplit(url)
        parts.port  # force lazy port parsing; invalid ports raise here
    except ValueError as exc:
        raise ConfigError(f"{provider_id} base URL 형식이 올바르지 않습니다.") from exc
    host = (parts.hostname or "").lower()
    if parts.scheme not in {"https", "http"} or not host:
        raise ConfigError(
            f"{provider_id} base URL이 올바르지 않습니다: {_safe_url_display(url, parts)} "
            "(https://host 형식이어야 합니다)"
        )
    loopback = _is_loopback_host(host)
    if parts.scheme == "http" and not loopback:
        raise ConfigError(
            f"{provider_id} base URL은 https여야 합니다: {_safe_url_display(url, parts)} "
            "(http는 루프백 주소만 허용됩니다)"
        )
    if not loopback and host not in ALLOWED_PROVIDER_HOSTS.get(provider_id, set()):
        if not allow_custom_endpoint:
            raise ConfigError(
                f"{provider_id} base URL 호스트가 허용 목록에 없습니다: {host} "
                f"(허용: {', '.join(sorted(ALLOWED_PROVIDER_HOSTS.get(provider_id, set())))}; "
                "신뢰할 수 있는 엔드포인트라면 --allow-custom-endpoint를 사용하세요)"
            )
        message = f"{provider_id} 사용자 지정 엔드포인트를 허용했습니다: {_safe_url_display(url, parts)}"
        if emitter is not None and hasattr(emitter, "emit"):
            emitter.emit("config_warning", message=message)
        else:
            print(f"경고: {message}", file=sys.stderr)
    return url.rstrip("/")


def provider_candidates(
    environment: Mapping[str, str] | None = None,
    provider: str = "",
    *,
    allow_custom_endpoint: bool = False,
    emitter=None,
) -> list[ProviderConfig]:
    """Provider candidates in fixed order (OpenRouter → OpenCode Go).

    Every candidate base_url is validated: https only (http solely for
    loopback dev gateways) and the host must be allowlisted unless
    allow_custom_endpoint / MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT opts in.
    """
    env = dict(load_environment() if environment is None else environment)
    allow_custom = allow_custom_endpoint or _bool_value(env.get("MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT"), False)
    return [
        ProviderConfig(
            id=spec["id"],
            label=spec["label"],
            api_key=spec["api_key"],
            base_url=_validate_base_url(
                spec["id"],
                spec["base_url"],
                allow_custom_endpoint=allow_custom,
                emitter=emitter,
            ),
            model=spec["model"],
        )
        for spec in _candidate_specs(env, provider)
    ]


def _candidate_specs(env: Mapping[str, str], provider: str = "") -> list[dict[str, str]]:
    """Provider selection without endpoint validation: ``[{id, label,
    api_key, base_url, model}]`` in fixed order (OpenRouter → OpenCode Go).

    Single source for candidate identity — ``provider_candidates`` builds
    validated configs from these, and ``summary_config_fingerprint`` hashes
    them so the *effective* provider set (which keys are present, which
    endpoints/models they resolve to) is part of the skip decision.
    """
    requested = str(provider or "").strip().lower()
    specs: list[dict[str, str]] = []
    openrouter_key = str(env.get("OPENROUTER_API_KEY", "")).strip()
    opencode_key = str(env.get("OPENCODE_API_KEY", "")).strip()
    if openrouter_key and requested != "opencode":
        specs.append(
            {
                "id": "openrouter",
                "label": "OpenRouter",
                "api_key": openrouter_key,
                "base_url": str(env.get("OPENROUTER_BASE_URL", "") or DEFAULT_OPENROUTER_BASE_URL),
                "model": str(env.get("OPENROUTER_MODEL", "") or DEFAULT_OPENROUTER_MODEL),
            }
        )
    if opencode_key and requested != "openrouter":
        specs.append(
            {
                "id": "opencode",
                "label": "OpenCode Go",
                "api_key": opencode_key,
                "base_url": str(env.get("OPENCODE_API_BASE_URL", "") or DEFAULT_OPENCODE_BASE_URL),
                "model": str(env.get("OPENCODE_MODEL", "") or DEFAULT_OPENCODE_MODEL),
            }
        )
    return specs
