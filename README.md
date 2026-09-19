# MediaScribe-CLI

> 회의·강의·인터뷰 녹음을 명령행 한 줄로 **화자 스크립트 + AI 요약 + 전체 전사문**으로 바꾸는 도구

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-206%20passing-brightgreen)](#개발-검증)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)]()

MediaScribe 데스크톱 UI와 별개로 실행되는 Python CLI입니다.
설계 문서는 [`DESIGN.md`](./DESIGN.md)에 있습니다.

```bash
mediascribe transcribe ./meeting.m4a --output-dir ./outputs --language ko --format md
```

```text
[meeting.m4a] STT 42.0% | 경과 00:18 | 예상 잔여 00:24 | 미디어 스크라이브 회의
```

```text
outputs/
├─ meeting_화자스크립트.md   # [시작–끝] 화자: 발화
├─ meeting_본문요약.md       # AI 요약 (결정·액션·미확정)
└─ meeting_본문전체.md       # 용어 사전 적용 전체 전사문
```

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 🎙️ 전사 | faster-whisper STT, 단일/일괄 처리, 실시간 진행률·경과·ETA |
| 👥 화자 분리 | 선택적 pyannote.audio, 화자 레이블 매핑 |
| 🤖 AI 요약 | OpenRouter 우선 → OpenCode Go fallback, 긴 전사는 map-reduce 청킹 |
| 📄 3종 산출물 | 입력당 화자 스크립트·본문 요약·본문 전체본 정확히 3개 (html/md/txt) |
| 🔁 재개 | 매니페스트 + 체크포인트 — 중단 후 재실행 시 완료된 작업 건너뜀 |
| 🔒 원자 커밋 | 실패·중단 시 부분 산출물 0개, 이전 유효 커밋은 digest 검증 후 보존 |
| 🛡️ 보안 | 키는 환경 변수만, 허용 목록 엔드포인트만, symlink 입력 거부, 로그 비밀 마스킹 |

## 📦 설치

Python 3.11 이상을 권장합니다.

```bash
cd MediaScribe-CLI
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

화자 분리가 필요하면 추가 의존성을 설치합니다.

```bash
python -m pip install -e '.[diarization]'
```

```bash
mediascribe --help
# 또는
python -m mediascribe_cli --help
```

## 🚀 기본 실행

```bash
mediascribe transcribe ./meeting.m4a \
  --output-dir ./outputs \
  --language ko \
  --format md
```

`--format`은 `html`, `md`, `txt` 중 하나입니다. 기본값은 `md`입니다.

### 산출물 의미

- **화자 스크립트**: `[시작–끝] 화자: 발화` 형식의 대화 스크립트
- **본문 요약본**: 전사에 근거한 AI 요약. 결정사항·후속 액션·미확정 사항 포함
- **본문 전체본**: 용어 사전이 적용된 전체 전사문

HTML은 외부 CSS나 JavaScript 없이 열 수 있는 self-contained 문서입니다.

### 원자적 커밋과 실패 동작

세 산출물은 임시 파일에 먼저 기록된 뒤 `os.replace`로 한꺼번에 커밋됩니다.
전사·요약·저장 중 어느 단계에서든 실패하거나 중단(Ctrl+C)되면 해당 실행의
부분 산출물은 남지 않습니다. 단, **이전 실행이 성공적으로 커밋한 완전한
세트는 보존됩니다** — 보존 여부는 커밋 시 기록된 sha256 digest로 검증되므로
부분적으로 교체되거나 편집된 혼합 세트는 유효 산출물로 남지 않습니다.
다른 포맷으로 다시 실행하면(예: md → html) 이전 포맷 파일은 새 산출물
커밋이 성공한 뒤에만 정리됩니다. `--no-resume`은 매니페스트가 없어 digest
검증이 불가하므로 디스크의 완전한 세트를 존재 여부로만 보존합니다.

### 출력 이름 충돌

같은 이름(stem)의 입력이 둘 이상이면 확장자로 구분합니다.

```text
meeting.m4a + meeting.wav →
  meeting.m4a_화자스크립트.md, meeting.m4a_본문요약.md, meeting.m4a_본문전체.md
  meeting.wav_화자스크립트.md, meeting.wav_본문요약.md, meeting.wav_본문전체.md
```

다른 디렉터리의 완전히 같은 파일명은 경로 해시가 붙어 구분됩니다. 기존
산출물을 조용히 덮어쓰는 일은 없습니다.

> **알려진 동작(수용됨)**: 이미 처리된 입력과 같은 stem의 파일을 나중에
> 추가하면 출력 키가 바뀝니다(예: `meeting` → `meeting.m4a`). 이전 키로
> 커밋된 산출물 세트는 고아가 되어 자동으로 정리되지 않습니다 — 필요하면
> 이전 이름의 파일을 수동으로 삭제하세요.
>
> **입력 바인딩**: 매니페스트는 각 키를 입력 파일의 실제 경로에 연결합니다.
> 다른 디렉터리의 같은-stem 파일(예: `2024-01/meeting.m4a` 다음
> `2024-02/meeting.m4a`)은 `meeting.m4a.<hash>` 키로 구분되어 이전 산출물을
> 덮어쓰지 않습니다.

## 🔑 환경 설정(명시적)

API 키와 설정은 다음 순서로 읽습니다(먼저 찾은 값이 우선).

1. 프로세스 환경 변수
2. `--env-file <경로>` 또는 `MEDIASCRIBE_ENV_FILE=<경로>`
3. `~/.config/mediascribe/env` (사용자 수준 설정)

**작업 디렉터리의 `.env`/`.env.local`은 읽지 않습니다.** 적대적인 디렉터리가
키와 전사를 다른 곳으로 보내는 것을 막기 위한 보안 정책입니다. 기존
`.env.local`을 쓰던 경우 다음 중 하나를 선택하세요.

```bash
mediascribe transcribe ./meeting.m4a --env-file .env.local ...
# 또는
export MEDIASCRIBE_ENV_FILE=.env.local
# 또는
cp .env.local ~/.config/mediascribe/env && chmod 600 ~/.config/mediascribe/env
```

키가 들어 있는 파일은 `chmod 600`으로 제한하세요. 그룹/다른 사용자가 읽을
수 있는 파일에서 비밀 값을 읽으면 stderr에 경고가 표시됩니다.

## 📊 실시간 진행률

일반 터미널에서는 stderr에 현재 상태가 갱신됩니다.

```text
[meeting.m4a] STT 42.0% | 경과 00:18 | 예상 잔여 00:24 | 미디어 스크라이브 회의
```

표시 단계:

- 모델 로딩
- STT 진행률
- 화자 분리
- 본문 요약 API 요청(구간별 진행률 포함)
- 파일별 완료

STT ETA는 faster-whisper가 제공하는 미디어 duration과 현재 세그먼트 위치를
기준으로 계산합니다. 모델 로딩·화자 분리·요약처럼 전체 작업량을 알 수 없는
단계는 진행률 대신 단계와 **실측 경과 시간**을 표시합니다.

자동화할 때는 JSONL 이벤트를 사용합니다.

```bash
mediascribe transcribe ./meeting.m4a \
  --format html \
  --json-lines > events.jsonl
```

`progress` 이벤트에는 `stage`, `progress`, `elapsed_seconds`, `eta_seconds`,
`text`가 포함됩니다. JSONL의 전사 텍스트는 **기본적으로** sha256 해시로
대체됩니다 — 원문이 필요하면 `--events-full-text`(또는
`MEDIASCRIBE_EVENTS_FULL_TEXT=1`)로 명시적으로 opt-out하세요. 사람용 stderr
표시는 그대로입니다.

## 🔁 재개와 체크포인트

실행 상태는 `<output-dir>/.mediascribe/` 아래에 저장됩니다(디렉터리 0700,
파일 0600 — 숨김 메타데이터이며 산출물 3개 계약에 포함되지 않습니다).

- `manifest.json`: 파일별 상태(pending → transcribing → transcribed →
  summarizing → committed/failed)와 산출물 경로
- `checkpoints/<key>.transcript.json`: 전사 결과 체크포인트
- `checkpoints/<key>.summary.json`: 긴 전사의 구간별 부분 요약 체크포인트
- `lock`: 동시 실행 방지 잠금(다른 실행이 같은 출력 디렉터리를 쓰면 종료 코드 2)

재실행하면 이미 `committed`된 입력은 건너뛰고(`file_skipped` 이벤트),
요약에서 실패한 입력은 전사를 다시 하지 않고 요약부터 재개합니다.
`--no-resume`(또는 `MEDIASCRIBE_RESUME=0`)은 매니페스트를 완전히 끄고 항상
새로 처리합니다 — 이 경우에도 산출물 커밋은 원자적입니다.

## 👥 화자 분리·용어 정제

Hugging Face에서 pyannote 모델 사용 약관을 승인한 뒤 `HF_TOKEN`을 설정합니다.
**`HF_TOKEN`은 환경 변수 또는 환경 파일로만 전달합니다** — 명령행 플래그가
없으므로 토큰이 `ps`/`/proc`이나 로그에 노출되지 않습니다. `--env-file`이나
`~/.config/mediascribe/env`에 둔 `HF_TOKEN`도 인식됩니다.

```bash
export HF_TOKEN="hf_..."
mediascribe transcribe ./meeting.m4a \
  --output-dir ./outputs \
  --format md \
  --language ko \
  --diarize \
  --num-speakers 2 \
  --speaker-labels "A,B" \
  --glossary examples/glossary.json \
  --speaker-map examples/speakers.json
```

용어 사전 예시는 `examples/glossary.json`, 화자 매핑 예시는
`examples/speakers.json`입니다.

## 🤖 AI 본문 요약

환경 파일 또는 프로세스 환경 변수에 API 키를 설정합니다. API 키는 명령행
인자로 전달하지 않습니다.

```bash
cp .env.example ~/.config/mediascribe/env
chmod 600 ~/.config/mediascribe/env
# 파일에 OPENROUTER_API_KEY 입력
```

공급자 순서:

1. OpenRouter — `deepseek/deepseek-v4-flash-0731:free`
2. OpenCode Go — `deepseek-v4-flash` fallback

OpenRouter가 실패하면 `OPENCODE_API_KEY`가 설정된 경우 OpenCode Go로
재시도합니다. 두 공급자 모두 없으면 본문 요약 단계에서 오류 코드 `4`로
종료합니다.

### 엔드포인트 보안 정책

API 키와 전사는 허용 목록의 https 호스트로만 전송됩니다.

- `openrouter.ai`, `opencode.ai`만 기본 허용
- `http://`는 루프백(127.0.0.1, ::1, localhost) 개발용 게이트웨이만 허용
- 다른 https 엔드포인트는 `--allow-custom-endpoint` 또는
  `MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT=1`로 명시적 opt-in 필요(경고 이벤트 발생)
- 잘못된/비허용 엔드포인트는 종료 코드 `2`로 거부

### 긴 전사의 구간 요약

전사가 `--summary-chunk-chars`(기본 24,000자, 최대 256,000자)를 넘으면
구간별 부분 요약 후 통합 요약(map-reduce)을 수행합니다. 각 구간 요약은
체크포인트에 저장되어 중단 후 재개 시 완료된 구간을 다시 요청하지 않습니다.
모든 요청은 크기·시간 상한이 있고, 재시도 가능 오류는 공급자당 최대 2회
재시도 후 다음 공급자로 fallback합니다. 구간 수는 512개로 제한되며, 더 긴
전사는 `--summary-chunk-chars`를 늘려야 합니다.

이미 만든 전사 파일에서 요약만 생성할 수도 있습니다.

```bash
mediascribe minutes ./outputs/meeting_화자스크립트.md \
  --format html \
  --output ./outputs/meeting_본문요약.html \
  --title "제품 회의"
```

`minutes`도 `--env-file`, `--allow-custom-endpoint`, `--summary-chunk-chars`,
`--events-full-text`를 지원하며 출력은 원자적으로 기록됩니다. 입력 전사는
32 MiB까지 허용됩니다. 심볼릭 링크 입력은 미디어와 같은 기본 거부 정책으로
거부됩니다.

## 🧪 검증만 하기(--dry-run)

```bash
mediascribe transcribe ./recordings --output-dir ./outputs --dry-run
```

모델을 로드하지 않고 입력 수집, 출력 키(이름 충돌 결과 포함), 파일별
매니페스트 상태(`will_skip`/`will_process`), 요약 구간 수 추정, 검증된
공급자 엔드포인트를 보고합니다. 잠금·쓰기·`.mediascribe` 생성이 없습니다.

## 🚪 종료 코드

| 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `2` | 인자·설정·입력·잠금 오류 |
| `3` | ML 런타임 의존성 오류 |
| `4` | AI provider 본문 요약 오류 |
| `5` | 일부 입력 파일 실패(나머지 파일은 계속 처리) |
| `130` | Ctrl+C 중단 |

## 🛠️ 개발 검증

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q src tests
```

테스트는 무거운 ML 패키지를 실제로 내려받지 않고 fake runtime을 주입해 STT
진행 이벤트, 화자 분리, 세 산출물, 매니페스트 재개, provider fallback을
검증합니다.

### 실제 미디어 수동 스모크 게이트(릴리스 전)

```bash
# 30초 테스트 WAV 생성
ffmpeg -f lavfi -i "sine=frequency=440:duration=30" -ar 16000 /tmp/smoke.wav

mediascribe transcribe /tmp/smoke.wav \
  --output-dir /tmp/smoke-out \
  --language ko --format md
# 확인: 산출물 정확히 3개, 진행 줄에 0이 아닌 경과 시간

# 중단 테스트: STT 중 Ctrl+C → 최종 산출물 0개, 체크포인트 존재
# 재실행 → file_skipped/체크포인트 재사용으로 STT 없이 완료
```
