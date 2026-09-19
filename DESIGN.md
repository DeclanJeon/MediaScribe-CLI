# MediaScribe-CLI 설계

## 1. 목표

MediaScribe 데스크톱 UI와 독립적으로 음성·영상 전사, 화자 분리, 용어 정제, 본문 요약을 명령행에서 실행한다.

핵심 원칙:

- UI 프로젝트를 import하거나 실행하지 않는다.
- faster-whisper의 스트리밍 세그먼트를 터미널에 실시간 반영한다.
- 현재 진행률, 경과 시간, 예상 잔여 시간을 사람이 읽을 수 있게 표시한다.
- 최종 산출물은 입력 파일당 정확히 3개다.
- 사용자가 `html`, `md`, `txt` 중 하나의 포맷을 선택한다.
- OpenRouter를 먼저 시도하고, 실패하면 설정된 OpenCode Go로 fallback한다.
- 비밀키는 명령행 인자로 받지 않는다.
- 실패하거나 중단된 입력은 최종 산출물을 남기지 않는다(원자적 커밋).
- 재실행은 완료된 입력을 건너뛰고 체크포인트에서 재개한다.

## 2. 최종 산출물 계약

입력 파일이 `meeting.m4a`이고 `--format md`이면 다음 파일을 만든다.

```text
meeting_화자스크립트.md
meeting_본문요약.md
meeting_본문전체.md
```

### 화자 스크립트

화자 레이블, 구간 타임스탬프, 발화를 포함한다.

```text
[00:00:01–00:00:04] A: 안녕하세요.
[00:00:05–00:00:08] B: 반갑습니다.
```

### 본문 요약본

화자 스크립트를 근거로 OpenRouter/OpenCode Go가 생성한다. 전사에 없는 사실을 추가하지 않으며, 결정·액션·미확정 사항과 근거 시간을 포함한다.

### 본문 전체본

용어 사전이 적용된 전체 전사문이다. 원문은 내부 정규 데이터의 `raw_text`에 보존되고, 사용자 산출물에는 검수 가능한 정제 본문을 저장한다.

포맷별 표현:

- `txt`: 순수 텍스트
- `md`: 제목과 구간 목록을 포함한 Markdown(전사 텍스트는 마크업 이스케이프)
- `html`: 외부 의존성 없는 self-contained HTML

### 원자적 커밋

세 산출물은 `ArtifactStager`가 `<final>.tmp-<pid>-<kind>` 임시 파일(0600, fsync)에 기록한 뒤 `os.replace`로 커밋한다. 커밋 순서는 저널 마커(`.mediascribe-commit-<pid>`, 이름+sha256 기록) → rename ×3 → 디렉터리 fsync → 마커 삭제 → 이전 포맷 stale 파일 정리다. 실패·중단 시 `abort()`가 임시 파일을 지운다. **이전 실행이 커밋한 완전한 세트는 보존한다** — 보존 여부는 기록된 `output_digests`의 sha256 검증으로 판단하므로, 부분적으로 교체되거나 편집된 혼합 세트는 유효 커밋으로 오인되지 않고 제거된다(이번 실행의 부분 산출물만 정리). `--no-resume`은 매니페스트가 없어 digest 검증이 불가하므로 디스크의 완전한 세트를 존재 여부로만 보존한다. stale 정리는 3종 커밋에서만 적용되며, `minutes --output` 같은 단일 산출물 커밋은 같은 stem의 무관한 파일을 절대 삭제하지 않는다.

### 출력 키와 이름 충돌

산출물 stem은 `output_key()`가 결정한다. stem이 입력 전체에서 유일하면 `meeting`, 같은 stem이 둘 이상이면 `meeting.m4a`처럼 확장자를 붙이고, 파일명까지 같으면 경로 해시를 붙인다. 조용한 덮어쓰기는 없다. **매니페스트 엔트리는 입력의 resolved 경로(`media_path`)에 바인딩된다** — 다른 경로의 같은-stem 파일이 같은 키에 오면 `meeting.m4a.<hash8>` 형태로 키를 구분하고 `config_warning`을 낸다. 이미 처리된 입력과 같은 stem의 파일을 나중에 추가하면 키가 바뀌어 이전 산출물 세트가 고아가 된다 — 수용된 동작이며 수동 정리가 필요하다.

## 3. 프로젝트 구조

```text
MediaScribe-CLI/
├─ pyproject.toml
├─ README.md
├─ DESIGN.md
├─ .env.example
├─ .gitignore
├─ examples/
├─ src/mediascribe_cli/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ cli.py              # argparse 명령과 종료 코드, 얇은 orchestration
│  ├─ pipeline.py         # 파일별 상태 기계: STT → 체크포인트 → 요약 → 커밋
│  ├─ manifest.py         # 실행 매니페스트, 파일 상태, 체크포인트, 잠금
│  ├─ artifacts.py        # 출력 키, 스테이징, 원자적 커밋, stale 정리
│  ├─ chunking.py         # 결정적 전사 청킹(문자 예산)
│  ├─ config.py           # 환경 변수·env 파일·CLI 옵션 병합, 엔드포인트 정책
│  ├─ events.py           # 실시간 터미널 표시와 JSONL 이벤트
│  ├─ errors.py           # 안정적인 종료 코드
│  ├─ transcriber.py      # faster-whisper + pyannote (산출물 쓰기 없음)
│  ├─ transcript.py       # 정제·렌더링·transcript_lines
│  └─ meeting_minutes.py  # bounded HTTP + retry/fallback + 청크 요약
└─ tests/
```

의존 방향(사이클 없음): `cli → {pipeline, meeting_minutes}`; `pipeline → {transcriber, meeting_minutes, artifacts, manifest, chunking, transcript}`; `meeting_minutes → {config, chunking, errors, events}`; `transcriber → {config, transcript}`; `transcript → {artifacts, chunking}`; `config → {artifacts, chunking}`; `manifest`/`artifacts`/`chunking`은 stdlib + `errors`만 사용한다.

## 4. 명령 계약

### 전사·세 산출물 생성

```bash
python -m mediascribe_cli transcribe INPUT \
  --output-dir ./outputs \
  --format md \
  --language ko \
  --diarize \
  --num-speakers 2 \
  --glossary glossary.json \
  --speaker-map speakers.json \
  --meeting-title "제품 회의"
```

- `INPUT`: 파일 또는 디렉터리
- `--format`: `html`, `md`, `txt` 중 하나. 기본값 `md`
- `--json-lines`: stdout에 구조화 이벤트를 출력하고 일반 진행 로그는 stderr로 보낸다.
- `--dry-run`: 입력·설정만 검증하고 모델을 로드하지 않는다. 출력 키, 파일별 매니페스트 상태, 요약 구간 수 추정, 검증된 공급자 엔드포인트를 보고한다. 잠금·쓰기 없음.
- `--env-file`: 명시적 환경 파일(`MEDIASCRIBE_ENV_FILE`과 동일)
- `--allow-custom-endpoint`: 허용 목록 외 https 공급자 엔드포인트 opt-in
- `--no-resume`: 매니페스트·체크포인트 재개 비활성화
- `--follow-symlinks`: 입력 디렉터리 안의 심볼릭 링크 추적(밖으로 나가는 링크는 거부)
- `--summary-chunk-chars`: 요약 요청 한 번의 최대 전사 문자 수
- `--events-full-text`: JSONL 이벤트의 전사 텍스트 해시 대체를 끄고 원문을 포함(기본은 해시 대체)

### 기존 전사에서 요약만 생성

```bash
python -m mediascribe_cli minutes transcript_speaker.txt \
  --format html \
  --output meeting_본문요약.html \
  --title "제품 회의"
```

입력은 32 MiB까지 허용되고, 출력은 `ArtifactStager`로 원자적으로 기록된다. `--env-file`/`--allow-custom-endpoint`/`--summary-chunk-chars`/`--events-full-text`를 지원한다. 심볼릭 링크 입력은 미디어와 같은 기본 거부 정책으로 거부된다(`--follow-symlinks` 없음).

## 5. 파이프라인 데이터 흐름

```text
collect_media_files ──► output_key ──► manifest.should_skip? ──yes──► file_skipped
        │no
        ▼
transcribe_file (STT 이벤트) ──► save_transcript_checkpoint ──► state=transcribed
        ▼
transcript_lines ──► chunk_transcript ──► 구간별 _request_with_retry
        │                                   (digest별 체크포인트)
        ▼
_reduce ──► GeneratedMinutes
        ▼
ArtifactStager.stage ×3 ──► commit (rename → fsync → stale 정리)
        ▼
state=committed ──► file_complete
```

어떤 예외든 `stager.abort()` + 이번 실행의 부분 산출물 정리(이전 커밋 세트는 보존) + `state=failed` + `file_failed`가 실행된다. 체크포인트는 남아 다음 실행이 `transcribed` 또는 실패한 구간 digest부터 재개한다.

매니페스트 상태 기계: `pending → transcribing → transcribed → summarizing → committed`, 실패 시 `failed`(재시도 가능). `should_skip`은 `committed` + 파일·설정 핑거프린트 일치 + 요약 핑거프린트(제목·**유효 공급자 후보 집합**(어떤 키가 있고 각각 어떤 엔드포인트/모델로 resolve되는지 — 키 값 자체는 해시하지 않음)·청크 크기) 일치 + 기록된 산출물이 정확히 세 종류이고 각각 출력 디렉터리 안의 결정적 경로에 있는 심볼릭 링크 아닌 실제 파일이며 `output_digests`와 일치할 때만 참이다 — 제목·공급자·모델이 바뀐 재실행은 재요약한다. 전사 체크포인트는 저장된 `media_path`가 현재 파일로 resolve되고 파일·전사-설정 핑거프린트(`artifact_format` 제외)가 모두 일치할 때만 재사용된다 — 포맷만 바꾼 재실행은 STT 없이 재렌더링한다. 체크포인트 세그먼트는 유한 숫자 `start ≤ end` 형태 검증을 통과해야 하며, malformed는 missing 취급이다. 실행 시작 시 `committed`가 아닌 모든 엔트리의 부분 산출물을 정리하되, 기록된 산출물 세트가 digest 검증을 통과한 완전 세트면(이전 실행의 유효 커밋) 보존한다. `media_path`가 없는 레거시 엔트리는 바인딩 증명이 불가하므로 다른 입력 취급 — 해시 키로 구분한다.

## 6. 실시간 진행률과 ETA

사람용 기본 출력은 stderr에서 갱신한다. 세그먼트가 생성될 때마다 다음 정보를 표시한다.

```text
[meeting.m4a] STT 42.0% | 경과 00:18 | 예상 잔여 00:24 | 미디어 스크라이브 회의
```

- faster-whisper `info.duration`과 현재 세그먼트 종료 시각으로 STT 진행률을 계산한다.
- 경과 시간은 파일별 `time.monotonic()` 기준이다.
- 진행률과 경과 시간으로 ETA를 계산한다.
- 모델 로딩·화자 분리·요약은 정확한 퍼센트를 알 수 없으므로 단계와 **실측 경과 시간**을 표시한다(완료 이벤트는 0이 아닌 경과를 담는다).
- 요약은 구간별로 `progress` 이벤트를 낸다.
- 터미널이 아닌 환경에서는 같은 이벤트를 일반 줄로 출력한다.
- 사람용 출력은 ANSI 이스케이프·제어 문자·비밀 패턴을 제거한다.

`--json-lines` 이벤트에는 다음 필드를 포함한다.

```json
{"type":"progress","stage":"stt","progress":42.0,"elapsed_seconds":18.2,"eta_seconds":24.0,"text":"..."}
```

주요 이벤트: `run_started`, `file_started`, `file_skipped`, `checkpoint_saved`, `progress`, `diarization_started`, `diarization_complete`, `summary_started`, `provider_failed`, `config_warning`, `minutes_complete`, `file_complete`, `file_failed`, `run_complete`, `error`.

`transcript_segment`/`progress`의 `text`는 **기본적으로** `{"sha256": …, "chars": n}`으로 대체된다. `MEDIASCRIBE_EVENTS_FULL_TEXT=1` 또는 `--events-full-text`로 opt-out하면 원문이 포함된다.

## 7. 환경 설정

우선순위는 다음과 같다(먼저 찾은 값이 우선).

1. 프로세스 환경 변수
2. `--env-file` 또는 `MEDIASCRIBE_ENV_FILE`
3. `~/.config/mediascribe/env`

작업 디렉터리와 패키지 디렉터리의 `.env*`는 **읽지 않는다**(적대적 cwd가 키·전사를 탈취하는 것을 방지). 비밀 값이 들어 있는 파일이 그룹/다른 사용자에게 읽히면 stderr 경고를 낸다.

회의록 provider 순서는 고정이다.

1. `OPENROUTER_API_KEY`
   - 기본 모델: `deepseek/deepseek-v4-flash-0731:free`
   - 기본 URL: `https://openrouter.ai/api/v1`
2. `OPENCODE_API_KEY`
   - 기본 모델: `deepseek-v4-flash`
   - 기본 URL: `https://opencode.ai/zen/go/v1`

OpenRouter가 키 없이 비활성화되었거나 요청이 실패하면 OpenCode Go가 설정된 경우에만 fallback한다. 두 provider가 모두 없으면 명확한 오류로 종료한다.

### 엔드포인트 정책

- https만 허용. http는 루프백(127.0.0.1, ::1, localhost)만 허용.
- 호스트는 `openrouter.ai`/`opencode.ai` 허용 목록에 있어야 한다. 다른 https 호스트는 `--allow-custom-endpoint`/`MEDIASCRIBE_ALLOW_CUSTOM_ENDPOINT=1` opt-in + `config_warning` 이벤트.
- 잘못된 URL은 `ConfigError`(종료 코드 2).

### 요청 경계

- 요청 본문 ≤ `4 × MAX_CHUNK_CHARS + 8 KiB`(UTF-8 최악 4바이트/문자 기준 — 청크는 문자 수로 자르지만 한도는 바이트로 강제), 응답 ≤ 1 MiB, 오류 본문 ≤ 64 KiB.
- timeout은 [5, 600]초로 클램프.
- 재시도 가능(네트워크, 408/409/425/429/5xx)은 공급자당 최대 2회, 그 다음 공급자로 fallback. 비재시도 4xx는 즉시 fallback. 30x 리다이렉트는 따르지 않고 `HTTPError`로 표면화된다 — Authorization이 허용 목록 밖 호스트로 전달되지 않는다.
- 전사가 `--summary-chunk-chars`(기본 24,000, 최대 256,000)를 넘으면 구간 요약 → 통합 요약(map-reduce, 최대 2라운드 + 결정적 절단). 구간 수는 512개로 제한되며 초과 시 `--summary-chunk-chars`를 늘리라는 오류로 거부한다. 구간별 부분 요약은 `sha256(provider|모델|제목|프롬프트 빌더|버전)[:16]:<chunk digest>` 네임스페이스 키로 체크포인트된다 — 제목·공급자·모델·프롬프트가 바뀐 재개는 오래된 부분 요약을 재사용하지 않는다.

## 8. 재개·체크포인트·잠금

실행 상태는 `<output-dir>/.mediascribe/`(0700, 파일 0600)에 저장된다.

- `manifest.json`: 파일별 상태·핑거프린트·산출물 경로·오류
- `checkpoints/<key>.transcript.json`: 전사 결과(저장된 media_path resolve + 파일·전사-설정 핑거프린트 + 세그먼트 형태 검증, malformed는 missing 취급)
- `checkpoints/<key>.summary.json`: 구간별 부분 요약(네임스페이스 digest 키)
- `lock`: `O_CREAT|O_EXCL` 잠금. 죽은 pid의 stale 잠금은 회수하고, 살아 있는 잠금은 `UsageError`(종료 코드 2). `--dry-run`은 잠금을 잡지 않는다.

`.mediascribe`와 `checkpoints`가 심볼릭 링크이거나 디렉터리가 아니면 `UsageError`로 거부한다(읽기·쓰기·dry-run 모두). 실행 시작 시 `committed`가 아닌 엔트리의 부분 산출물을 정리해 부분 커밋을 자가 치유하고, 삭제가 발생하면 `artifacts_removed` 이벤트를 낸다. 매니페스트 `version`이 다르거나 JSON이 깊게 중첩되어 파싱에 실패하면 corrupt처럼 quarantine한다.

`--no-resume`/`MEDIASCRIBE_RESUME=0`은 매니페스트를 만들지 않는다(원자적 커밋은 유지).

## 9. 내부 데이터 불변식

최종 파일은 3개지만 JSON 형태의 내부 데이터는 다음 정보를 유지한다.

- `segments[].raw_text`: 모델 원문
- `segments[].clean_text`: 용어 사전 적용 결과
- `segments[].text`: `clean_text` 호환 필드
- `speaker_id`, `speaker_label`: 화자 정보
- `words`, `confidence`: 화자 분리 시 단어 타임스탬프와 신뢰도

## 10. 종료 코드

- `0`: 전체 성공
- `2`: 인자·설정·입력·잠금 오류
- `3`: faster-whisper/pyannote 런타임 의존성 오류
- `4`: AI provider 요약 생성 오류
- `5`: 일부 입력 파일 실패(나머지 파일은 계속 처리)
- `130`: Ctrl+C 중단

## 11. 보안 정책

- 비밀키는 명령행 인자로 받지 않는다(`--hf-token` 없음; `HF_TOKEN`은 환경 변수만).
- API 키와 전사는 허용 목록 https 엔드포인트로만 전송된다.
- 작업 디렉터리 `.env*`는 신뢰하지 않는다.
- 사람용 로그는 제어 문자·ANSI·비밀 패턴을 제거하고, 오류 메시지에서 API 키를 마스킹한다.
- 산출물·메타데이터는 0600/0700 권한으로 기록된다.
- 심볼릭 링크 입력은 기본 거부, `--follow-symlinks`도 입력 디렉터리 내부로 제한한다. 미디어와 `minutes` 전사 입력은 `safeio.open_no_symlinks`로 연다 — 경로의 모든 구성 요소를 `O_NOFOLLOW`로 열어 수집-사용 간 symlink 교체(마지막 구성 요소와 상위 디렉터리 모두)를 차단한다. rename 교체(실제 디렉터리를 다른 실제 디렉터리로)는 `openat2` 없이 막을 수 없어 수용된 잔여 gap이다. 환경 파일·용어 사전·화자 매핑·상태 JSON은 크기 상한이 있다.

## 12. 검증 전략

- ML 패키지를 설치하지 않아도 CLI parsing/config/transcript/provider 테스트가 실행되어야 한다.
- fake faster-whisper/pyannote 모듈로 실제 모델 없이 스트리밍 진행률과 세 산출물 계약을 검증한다.
- 파이프라인 통합 테스트는 매니페스트 재개, 체크포인트 재사용, 같은-stem 충돌, 포맷 재실행, 실패 시 산출물 0개를 검증한다.
- provider fallback은 네트워크 mock으로 OpenRouter 실패 후 OpenCode 호출 순서를 검증한다.
- 실제 API 호출과 실제 미디어는 기본 테스트에 포함하지 않는다 — 릴리스 전 수동 스모크 게이트(README 참조)로 검증한다.
