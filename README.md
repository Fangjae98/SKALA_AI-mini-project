# SKALA-AI_mini-project
# der_ops_agent — 분산에너지 운영 에이전트

태양광·ESS·변압기·EV 충전기 같은 분산에너지 설비의 상태를 **설비별 기준으로 진단**하고, **상황별 대응 매뉴얼**을 근거로 조치를 제안하며,
설비를 움직이는 조치는 **운영자가 승인·수정**한 뒤에만 반영하는 AI 운영 보조 웹 서비스입니다.

> SKALA AI Mini-Project 과제 2번(신규 프로젝트). 강의 제공 Base(IT 장애 진단 에이전트)의 뼈대를 분산에너지 운영으로 옮겨 다시 설계했습니다.

**문서는 4개입니다**

| 문서 | 내용 |
|---|---|
| **README.md** (이 문서) | 서비스 소개, 주요 기능, 실행 방법, 구조, 평가 기준 반영 |
| [운영_가이드.md](운영_가이드.md) | 설비 데이터·상태 판단 기준·대응 매뉴얼을 고치는 법 |
| [설계_결정_기록.md](설계_결정_기록.md) | 왜 이렇게 설계했나 (ADR 13건 + 캡스톤 강의 기준 개선 계획) |
| [실행_측정_기록.md](실행_측정_기록.md) | 실제로 돌려 본 테스트·측정 결과와 해석 |

---

## 1. 무엇을 하나 — 질문 3종류

| 질문 종류 | 예시 | 처리 |
|---|---|---|
| **잡담·안내** | 안녕 / 뭘 할 수 있어? / 어떤 매뉴얼이 있어? | 인사·도움말·매뉴얼 목록은 **AI 없이 코드가** 데이터로 바로 답함. 그 밖의 잡담은 AI가 짧게 |
| **설비 조회** | 어떤 설비가 있어? / 태양광 2호기 상태 알려줘 | 계측값 → **기준 대비 판정** → **문제점 분석** → 대응 질문 제안 (코드가 계산) |
| **이상 대응** | ESS 2호기 SOC가 바닥이에요 / ESS실에서 타는 냄새가 나요 | 전문 에이전트 조사 → 상황별 매뉴얼 검색(RAG) → 보고서 → 검수(코드+AI 심판) → **운영자 승인·수정** |

증상 단어(올라가요·바닥·경보·냄새 …)가 있으면 항상 이상 대응으로 보냅니다(`config/rules.json` → `intent`).

## 2. 주요 기능

| # | 기능 | 하는 일 | 파일 |
|---|---|---|---|
| 1 | **설비 현황 패널** | 화면 왼쪽에 설비 9대를 단계별로 표시(정상/주의/이상 + 대표 문제). 30초마다·답변 후 자동 새로고침. [상태 보기]·[대응 요청] 버튼 | `web_chat.html`, `server.py`(/equipment) |
| 2 | **설비별 상태 판단 기준** | 계측값을 설비 종류별 기준(주의·경보)과 비교해 상태·알람을 **코드가** 만듦. 데이터 파일에 알람을 손으로 적지 않음 | `agent/diagnosis.py`, `rules.json`(status_criteria) |
| 3 | **상태 분석 4단계** | ① 계측값 ② 기준 대비 판정 ③ 문제점 분석(의미 + 의심 이상) ④ 대응 질문 자동 생성 | `agent/intent.py`, `graph.py`(equipment) |
| 4 | **전문 에이전트 병렬 조사** | 전력 흐름 5단계를 4개 에이전트가 나눠 동시에 조회·점검 + 전력 수지 점검 | `agent/specialists.py` |
| 5 | **상황별 매뉴얼 RAG** | 매뉴얼 9종 × 상황 38개를 절 단위로 벡터화. 매뉴얼은 질문 의미 + 데이터 근거로, 상황은 키워드 → 의미 점수 → 기본 절차 순으로 선택 | `agent/rag.py`, `agent/selection.py`, `data/runbooks/` |
| 6 | **데이터 불일치 처리** | 운영자가 말한 이상이 데이터로 확인되지 않으면 설비를 멈추는 조치 대신 "데이터상 정상 — 현장·센서 확인" 안내. 단, 연기·냄새 같은 직접 관찰은 비상 대응 | `graph.py`(replan_check), `prompts.py` |
| 7 | **이중 검수** | 코드 검사(형식·지어낸 숫자) → 통과한 초안만 별도 AI 심판(4기준) | `agent/validation.py`, `graph.py`(validate) |
| 8 | **운영자 승인·수정** | 조치별로 골라서·문구를 고쳐서 승인 / **운영자 판단을 적으면 문제를 처음부터 재분석**(최신 데이터 재조회 → 매뉴얼 재선택 → 새 해결책 → 재검수 → 재승인) / 거부 | `graph.py`(approval), `agent/approval.py`, `web_chat.html` |
| 9 | **장애에도 멈추지 않음** | 설비·매뉴얼 조회가 일부 실패해도 부분 보고 + 받지 못한 데이터 명시, 조회 실패는 통신 두절 의심으로 게이트웨이까지 조사 | `agent/retry.py`, `specialists.py` |
| 10 | **운영 기반** | 승인 대기 SQLite 저장(재시작해도 이어짐), 대화방 분리, POST SSE 스트리밍(검수 통과한 답만 표시), 오류 시 화면 안내, JSON 로그 | `persistence.py`, `server.py`, `logs.py` |

## 3. 실행 방법

| 순서 | 명령 | 비용 | 용도 |
|---|---|---|---|
| 1 | `bash sh/setup_venv.sh` | 없음 | `.venv` 생성 + 패키지 설치 (최초 1회) |
| 2 | `cp .env.example .env` 후 키 입력 | 없음 | LangSmith·OpenAI 키 (Ollama만 쓰면 LangSmith 키만) |
| 3-a | `bash sh/setup_ollama.sh` → `bash sh/run_agent_ollma.sh` | 없음 | **기본 시연** — 로컬 실모델 (qwen3:0.6b 분류·계획 / qwen3:8b 작성·잡담 / bge-m3 임베딩, 생각 모드 끔) |
| 3-b | `bash sh/run_agent_openai.sh` | **과금** | OpenAI (gpt-4o-mini / gpt-4o, 심판 gpt-4o) |
| 3-c | `bash sh/run_agent_fake.sh` | 없음 | 가짜 모델(항상 같은 답) — 테스트·흐름 확인용 |
| 4 | `bash sh/run_frontend.sh` (새 터미널) | — | http://localhost:8600 |
| 5 | `.venv/bin/python -m pytest -q` | 없음 | 단위 테스트 83개 (약 45초) |
| 6 | `bash sh/run_eval.sh <fake\|ollama\|openai> <이름>` | 백엔드에 따름 | 질문 세트 측정 (기본 10 + 처음 보는 6) |

> 코드를 바꾸면 **에이전트 서버(8100)를 재시작**해야 합니다. 화면 서버(8600)만 다시 켜면 새 화면 + 옛 서버가 되어 기능이 엇갈립니다(예: '판단 반영'이 거부로 처리됨).

**환경변수**

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `LLM_BACKEND` | (스크립트가 지정) | `ollama` / `openai` / `fake` |
| `DER_MONITORING_FILE` | `data/monitoring.json` | 설비 데이터 파일. 테스트·평가는 `data/scenarios/full_incident.json` |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | 잡담 답변 모델 (0.6b는 엉뚱하게 답해서 8b 사용) |
| `OPENAI_JUDGE_MODEL` | `gpt-4o` | OpenAI 심판 모델 |
| `CHECKPOINTER` | `sqlite` | 승인 대기 저장소 (`memory`면 재시작 시 소실) |
| `DER_FAULT_INJECT` | (없음) | 예: `evc-01` — 이 설비 조회만 일부러 실패(부분 보고 시연) |
| `DER_LOG` / `DER_LOG_FILE` | `on` / (없음) | JSON 로그 끄기 / 파일 저장 |

## 4. 구조

```
브라우저(web_chat.html: 설비 현황 패널 + 채팅 + 승인 박스)
   └ web_demo_server.py :8600 ── server.py :8100  (/chat, /chat/stream, /chat/resume(/stream), /equipment)
                                    └ agent/graph.py (LangGraph Supervisor)

route ─┬ 잡담 → chat_answer           (인사·도움말·매뉴얼 목록은 코드가 바로)
       ├ 설비 조회 → equipment          (계측 → 기준 판정 → 문제점 분석)
       └ 이상 대응 → plan → gate → research(전문 에이전트 4개 병렬) → replan_check(상황별 RAG + 데이터 확인)
                    → analysis → write → validate(코드 + AI 심판) ─┬ 반려 → write (최대 3회)
                                                                    └ 통과 → approval(interrupt)
                                    approval ─┬ 승인(골라서·고쳐서)/거부 → finalize
                                              └ 운영자 판단 → research (재분석 루프)

mcp_server.py (FastMCP, stdio) — Tool: get_equipment_status · get_power_balance · search_runbooks
                                 Resource: runbook://der-response/{type}   Prompt: incident_analysis_prompt
config/rules.json — 판단 기준 한 곳 (질문 분류·상태 판단 기준·우선순위·RAG·승인 문구·재시도·동시성)
```

| 폴더/파일 | 역할 |
|---|---|
| `agent/graph.py` | 전체 흐름(노드·분기) |
| `agent/specialists.py` | 전문 에이전트 서브그래프 (collect → assess) |
| `agent/diagnosis.py` | 설비별 상태 판단·문제점 분석·대응 질문 |
| `agent/intent.py` | 질문 3분류, 코드 답변(인사·도움말·매뉴얼 목록·설비 현황) |
| `agent/rag.py`, `agent/selection.py` | 상황별 매뉴얼 검색, 매뉴얼 선택 규칙 |
| `agent/validation.py`, `agent/approval.py` | 숫자 검사, 승인 대상 조치 찾기 |
| `agent/prompts.py` | AI 지시문 한 곳 |
| `agent/{fake,ollama,openai}_llm.py` | AI 모델 3종 (같은 인터페이스) |
| `agent/deps.py` | 의존성 묶음(SupervisorDeps) |
| `agent/retry.py`, `limits.py`, `persistence.py`, `logs.py` | 재시도·동시성 상한·체크포인터 팩토리·로그 |
| `data/monitoring.json` | 시연용 설비 데이터 (이상 4대) |
| `data/scenarios/full_incident.json` | 테스트·평가용 고정 시나리오 (8대 이상) |
| `data/runbooks/*.md` | 대응 매뉴얼 9종 (기본 절차 + 상황 38개) |
| `tests/` | 단위 테스트 83개 |
| `eval/` | 질문 세트 측정, 운영 시나리오 측정, LangSmith 평가 |

## 5. 평가 기준은 어떻게 반영됐나

| 평가 항목 | 어디에 | 어떻게 |
|---|---|---|
| **Multi-Agent** | `agent/graph.py`, `agent/specialists.py` | Supervisor 그래프가 **단계별 전문 에이전트 4개**(데이터 수집·발전·계통·저장공급)에 조사를 나눠 맡김. 각 에이전트는 `collect → assess` **서브그래프**이고 `asyncio.gather`로 **동시에** 실행. 작성 모델과 별개인 **AI 심판**이 검수 담당(작성자 ≠ 검수자). 분류·계획은 작은 모델, 작성은 큰 모델로 역할 분리 |
| **MCP** | `mcp_server.py`, `agent/mcp_client.py` | FastMCP stdio 서버에 **Tool 3개**(설비 조회·전력 수지·매뉴얼 검색, 읽기 전용 표시), **Resource**(`runbook://der-response/{type}`, 수정 시각·해시 포함), **Prompt**(진단 절차). 클라이언트는 필요한 Resource를 가진 서버를 찾아 호출(방해용 서버 `mcp_server_docs.py` 포함), 서버별 동시 호출 상한, 실패는 표준 오류(`ToolError`)로 |
| **RAG** | `agent/rag.py`, `agent/selection.py`, `data/runbooks/` | 매뉴얼 9종을 **기본 부분 + 상황 절(38개) 단위로 벡터화**(임베딩: bge-m3 / OpenAI / TF-IDF). 매뉴얼 = 질문 의미 점수 + **데이터 근거 가산점** + 안전 우선순위, 상황 = 키워드 → 의미 점수 → 기본 절차. 보고서에는 **고른 상황 절만** 넘겨 엉뚱한 조치가 섞이지 않게 함 |
| **의존성 주입** | `agent/deps.py`, `graph.py`(`build_graph(deps)`) | `SupervisorDeps(fast_llm, quality_llm, mcp, session_store, version)`를 밖에서 만들어 노드에 `partial(..., deps=deps)`로 주입. 그래서 **모델 3종을 환경변수 하나로 교체**하고, 테스트에서는 가짜 모델·실패하는 MCP 대역·가짜 Ollama 서버를 끼워 넣음. 저장소도 팩토리(`create_checkpointer`)로 주입 |
| **LangSmith** | `@traceable` 42곳, `eval/langsmith_eval.py` | 노드·모델 호출·도구 호출을 추적(백엔드별 프로젝트 분리). **Dataset + aevaluate**로 혼합 평가: 규칙 평가 5종(판단·매뉴얼·안전 안내·미승인 조치·근거 없는 수치) + **AI 심판**. fake·Ollama·OpenAI 실험 3개 비교 |
| **모듈화** | `agent/` 22개 모듈, `config/rules.json`, `prompts.py` | 판단 기준(질문 분류·상태 기준·우선순위·RAG·승인 문구)을 **설정 파일 한 곳**, AI 지시문을 **한 파일**에. 진단·분류·검색·검수·승인·재시도·저장을 역할별 모듈로 분리. 모델 3종은 같은 인터페이스 |
| **단위 테스트** | `tests/` 11개 파일 **83개** | 설비 찾기·점검·상태 판단·질문 분류·상황별 매뉴얼·매뉴얼 선택·검수·승인(일부 승인·문구 수정·재분석)·장애 격리·재시작 후 승인·동시성. 외부 서비스 없이(가짜 모델 + 실제 MCP 서버 프로세스) 약 45초 |
| HITL (사람 승인) | `graph.py`(approval) | `interrupt()`로 멈추고 승인·부분 승인·문구 수정·**판단 반영 재분석**·거부. 대기 상태는 SQLite에 저장 |
| SSE 스트리밍 | `server.py`, `web_chat.html` | POST 스트리밍, 진행 단계 실시간 표시, 검수 전 초안 숨김, 오류 시 Error 이벤트 |

## 6. 버전 기록

| git 태그 | 내용 |
|---|---|
| `base` | 강의 제공 원본 (IT 서버 장애 진단) |
| `before` | 분산에너지로 주제만 옮긴 버전 |
| `after` | RAG·Multi-Agent·HITL·모듈화·검수 강화·단위 테스트 |
| (작업본) | 캡스톤 강의 기준 운영 안정성(I-1~I-12) + 질문 3분류·상태 판단 기준·상황별 매뉴얼·설비 현황 패널·데이터 불일치 처리·운영자 판단 재분석 |

## 7. 한계

- 처음 보는 표현에서 작은 계획 모델(qwen3:0.6b)이 조사할 설비를 잘못 고를 수 있음 → 질문에 나온 설비는 코드가 항상 포함하는 보강 필요
- 검수 3회 불합격 시 그대로 통과(무한 재작성 방지) → "검수 미통과" 표시 필요
- 질문에서 설비를 못 찾으면 기본 설비(연계점)를 조사 → 되묻기 필요
- MCP 호출에 시간 제한이 없음
- 판단 기준 숫자 대부분이 [가정] — 현장 기준으로 교체 필요
- 승인 대상은 놓치지 않도록 넓게 잡아서, 계획 문장("충전 계획 수립")도 승인을 물을 수 있음
