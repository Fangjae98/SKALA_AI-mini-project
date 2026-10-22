"""분산에너지 운영 에이전트 — FastAPI 백엔드 (Base all_total_vue와 동일 구조, Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨
- [After] HITL: 설비 제어 조치가 있으면 승인 대기 → POST /chat/resume 또는 /chat/resume/stream으로 재개

[개선 — 캡스톤 강의 기준, doc/설계_결정_기록.md 부록]
- I-3 체크포인터를 agent/persistence.py 팩토리로 만들어 그래프에 주입(기본 SQLite) — 승인 대기 중 서버를 재시작해도
      이어서 승인할 수 있다. 승인 대기 목록도 session.db에 저장한다. 끝난 실행의 체크포인트는 지운다.
- I-9 thread_id를 안 보내면 'default'를 같이 쓰던 문제 → 서버가 새 ID를 만들어 응답에 돌려준다.
      승인 대기 중인 대화방에 새 질문이 오면 앞의 승인 요청을 덮어쓰지 않고 "먼저 승인/거부" 안내를 돌려준다.
- I-7 POST /chat/stream·/chat/resume/stream 추가(질문이 URL에 남지 않음, 강의 방식). GET은 curl 확인용으로 유지.
      스트리밍 응답에 Cache-Control: no-cache, X-Accel-Buffering: no 헤더(프록시가 모아서 보내는 문제 방지).
- I-12 요청·그래프 이벤트를 JSON 한 줄 로그로(agent/logs.py), thread_id·run_id·langsmith_run_id로 연결.
"""
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.diagnosis import judge, problems, suggested_question
from agent.equipment import EQUIPMENT
from agent.graph import build_graph
from agent.intent import STAGE_ORDER, STATUS_LABELS
from agent.retry import ToolError, with_retry
from agent.logs import log_event, log_graph_event
from agent.mcp_client import MCPClient
from agent.persistence import create_checkpointer, forget_run
from agent.session_store import SqliteSessionStore
from agent.supervisor_v1 import run_v1

CREATION_COUNT = 0   # Wave 1 — 프로세스 내 Agent 번들 생성 횟수 (항상 1이어야 정상)
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}   # [개선 I-7]


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str
    thread_id: str | None = None   # [개선 I-9] 없으면 서버가 새로 만든다('default' 공유 금지)


class ResumeRequest(BaseModel):
    thread_id: str                 # [개선 I-9] 어느 대화방의 승인인지 반드시 지정
    decision: str = "approve"      # approve | reject | revise(운영자 판단을 반영해 다시 작성)
    comment: str = ""              # 운영자 판단·메모 (revise면 필수)
    actions: list[dict] | None = None   # [{original, text, checked}] — 일부만 승인하거나 문구를 고쳐 승인


def _build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015,
                             judge_model=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o"),   # [I-4 보정]
                             judge_usd_per_1k=0.005)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return backend, fast, quality


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CREATION_COUNT
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()   # Wave 2 — MCP subprocess는 서버 수명 동안 딱 1개
    backend, fast_llm, quality_llm = _build_llms()
    app.state.llm_backend = backend
    deps = SupervisorDeps(
        fast_llm=fast_llm,
        quality_llm=quality_llm,
        mcp=mcp,
        session_store=SqliteSessionStore(),
        version=os.environ.get("SUPERVISOR_VERSION", "v2"),
    )
    app.state.deps = deps
    app.state.instance_id = id(deps)
    async with create_checkpointer() as checkpointer:   # [개선 I-3] 저장소는 팩토리가, 그래프는 받기만
        app.state.checkpointer = checkpointer
        app.state.graph = build_graph(deps, checkpointer) if deps.version == "v2" else None
        log_event("ServerStarted", backend=backend, checkpointer=type(checkpointer).__name__,
                  supervisor_version=deps.version)
        yield
    await mcp.stop()
    deps.session_store.close()
    try:   # 종료 전 LangSmith 배치 전송 플러시 — kill 시 run이 pending으로 남는 것 방지
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT,
            "supervisor_version": app.state.deps.version,
            "llm_backend": app.state.llm_backend,
            "fast_llm": app.state.deps.fast_llm.name,
            "quality_llm": app.state.deps.quality_llm.name,
            "checkpointer": type(app.state.checkpointer).__name__}


@app.get("/equipment")
async def equipment_list():
    """화면 왼쪽 '설비 현황' 패널용 — 관리 중인 설비 전체의 현재 상태(MCP로 지금 값을 읽음).
    조회에 실패한 설비는 status='error'로 돌려준다(패널 전체가 비지 않게)."""
    mcp = app.state.deps.mcp

    async def one(eq_id: str) -> dict:
        try:
            d, _ = await with_retry(mcp.call_tool, "get_equipment_status", {"equipment_id": eq_id})
            findings = judge(d)   # 설비별 상태 판단 기준으로 진단 → 문제 수, 대응 질문(알람 복사가 아니라 진단 결과로)
            return {**d, "status_label": STATUS_LABELS.get(d["status"], d["status"]),
                    "problems": len(problems(findings)),
                    "suggested_question": suggested_question(d, findings)}
        except ToolError as e:
            info = EQUIPMENT.get(eq_id, {})
            return {"equipment_id": eq_id, "name": info.get("name", eq_id), "stage": info.get("stage", "-"),
                    "status": "error", "status_label": "조회 실패", "alerts": [e.user_message], "metrics": {},
                    "problems": 0, "suggested_question": f"{info.get('name', eq_id)}({eq_id}) 데이터가 안 들어와요. 원인과 대응 방법을 알려주세요."}

    items = await asyncio.gather(*(one(e) for e in EQUIPMENT))
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    items = sorted(items, key=lambda d: order.get(d.get("stage"), 99))
    summary = {label: sum(1 for d in items if d["status_label"] == label)
               for label in ("이상", "주의", "정상", "중지", "조회 실패")}
    return {"items": items, "summary": summary, "checked_at": time.strftime("%H:%M:%S")}


@app.get("/version")
async def version():
    return {"supervisor_version": app.state.deps.version}


def _metadata(result: dict, run_id: str | None) -> dict:
    return {
        "route": result.get("route"),
        "task_count": len(result.get("tasks", [])),
        "gate_removed": result.get("gate_removed", 0),
        "replan_rounds": result.get("replan_rounds", 0),
        "retry_count": result.get("retry_count", 0),
        "parallel_elapsed": result.get("parallel_elapsed"),
        "validation_attempts": result.get("validation_attempts", 0),
        "failures": result.get("failures", []),
        "cost_usd": result.get("cost_usd", 0.0),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


# ── [After 추가 — HITL] 승인 대기 관리 ─────────────────────────────────────
# 그래프 실행 1건마다 고유 run_id로 체크포인트를 따로 둔다(같은 대화방이라도 질문마다 상태가 섞이지 않게).
# [개선 I-3] 승인 대기 중인 실행은 대화방(thread_id)별로 session.db에 기억해 두고(재시작해도 유지),
# 승인/거부가 오면 이어서 실행한다. 끝난 실행의 체크포인트는 forget_run으로 지운다.
def _graph_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _new_thread_id(thread_id: str | None) -> str:
    return thread_id or f"t-{uuid.uuid4().hex[:12]}"


def _new_input(message: str, thread_id: str, history: list[dict]) -> dict:
    return {"query": message, "thread_id": thread_id, "history": history, "events": [], "cost_usd": 0.0}


def _interrupt_value(result_or_payload) -> dict | None:
    intr = result_or_payload.get("__interrupt__") if isinstance(result_or_payload, dict) else None
    return intr[0].value if intr else None


def _pending_notice(thread_id: str, pend: dict) -> dict:
    return {"status": "approval_pending", "thread_id": thread_id,
            "message": f"이 대화방에 승인을 기다리는 요청이 있습니다(\"{pend['message']}\"). "
                       "먼저 승인 또는 거부를 해 주세요.",
            "next": "POST /chat/resume {thread_id, decision: approve|reject}"}


def _log_result(result: dict, thread_id: str, run_id: str | None, started: float, ls_run: str | None) -> None:
    for ev in result.get("events", []):
        log_graph_event(ev, thread_id=thread_id, run_id=run_id)
    log_event("ChatFinished", thread_id=thread_id, run_id=run_id, langsmith_run_id=ls_run,
              route=result.get("route"), elapsed_ms=round((time.perf_counter() - started) * 1000),
              failures=len(result.get("failures", [])), cost_usd=result.get("cost_usd", 0.0))


@traceable(name="chat_request")
async def _handle_chat(message: str, thread_id: str) -> tuple[dict, str | None]:
    deps = app.state.deps
    started = time.perf_counter()
    history = deps.session_store.load(thread_id)   # Wave 12
    rt = get_current_run_tree()
    ls_run = str(rt.id) if rt else None
    run_id = None
    if deps.version == "v2":
        run_id = uuid.uuid4().hex
        result = await app.state.graph.ainvoke(_new_input(message, thread_id, history), _graph_config(run_id))
        approval = _interrupt_value(result)
        if approval:   # 승인 대기 — 대화 기록은 승인/거부 후 최종 답변과 함께 저장
            deps.session_store.set_pending(thread_id, run_id, message)
            log_event("ApprovalRequired", thread_id=thread_id, run_id=run_id, actions=len(approval["actions"]))
            return {**result, "approval_request": approval}, ls_run
        await forget_run(app.state.checkpointer, run_id)
    else:
        result = await run_v1(deps, message, thread_id, history)   # Wave 14
    deps.session_store.append(thread_id, "user", message)
    deps.session_store.append(thread_id, "assistant", result["final"])
    _log_result(result, thread_id, run_id, started, ls_run)
    return result, ls_run


@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = _new_thread_id(req.thread_id)
    if pend := app.state.deps.session_store.get_pending(thread_id):
        return _pending_notice(thread_id, pend)
    result, run_id = await _handle_chat(req.message, thread_id)
    if result.get("approval_request"):
        return {"status": "approval_required", "thread_id": thread_id, "draft": result.get("draft", ""),
                "approval": result["approval_request"], "events": result.get("events", []),
                "metadata": _metadata(result, run_id),
                "next": "POST /chat/resume {thread_id, decision: approve|reject}"}
    return {"status": "done", "thread_id": thread_id, "reply": result["final"],
            "events": result.get("events", []), "metadata": _metadata(result, run_id)}


@app.post("/chat/resume")
async def chat_resume(req: ResumeRequest):
    """[After 추가] 승인 대기 중인 실행을 승인(approve) 또는 거부(reject)로 이어서 끝낸다."""
    deps = app.state.deps
    pend = deps.session_store.get_pending(req.thread_id)
    if not pend:
        return {"status": "error", "message": f"thread_id={req.thread_id}에 승인 대기 중인 요청이 없습니다."}
    started = time.perf_counter()
    result = await app.state.graph.ainvoke(
        Command(resume=_resume_value(req.decision, req.comment, req.actions)), _graph_config(pend["run_id"]))
    if result.get("__interrupt__"):   # 수정 요청 → 다시 쓴 보고서로 또 승인 대기
        approval = result["__interrupt__"][0].value
        return {"status": "approval_required", "thread_id": req.thread_id, "draft": result.get("draft", ""),
                "approval": approval, "events": result.get("events", [])}
    deps.session_store.pop_pending(req.thread_id)
    await forget_run(app.state.checkpointer, pend["run_id"])
    deps.session_store.append(req.thread_id, "user", pend["message"])
    deps.session_store.append(req.thread_id, "assistant", result["final"])
    _log_result(result, req.thread_id, pend["run_id"], started, None)
    return {"status": "done", "thread_id": req.thread_id, "reply": result["final"],
            "events": result.get("events", []), "metadata": _metadata(result, None)}


async def _stream_run(graph_input, config: dict, seen: int, outcome: dict, thread_id: str):
    """그래프를 실행하며 SSE 문자열을 하나씩 내보낸다. 끝나면 outcome에 final / approval을 채운다.

    stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는 노드가 끝날 때마다의 상태 델타,
    "custom"은 write_node/chat_answer_node가 get_stream_writer()로 실행 중에 흘려보내는 토큰(AnswerToken)이다.
    승인 대기(interrupt)에 걸리면 updates에 '__interrupt__'가 들어온다."""
    run_id = config["configurable"]["thread_id"]
    async for mode, payload in app.state.graph.astream(graph_input, config,
                                                       stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield _sse(payload)
            continue
        approval = _interrupt_value(payload)
        if approval:
            outcome["approval"] = approval
            continue
        for node_name, delta in payload.items():
            if not isinstance(delta, dict):
                continue
            for ev in (delta.get("events") or [])[seen:]:
                log_graph_event(ev, thread_id=thread_id, run_id=run_id, node=node_name)
                yield _sse({**ev, "node": node_name})
            if delta.get("events"):
                seen = len(delta["events"])
            if delta.get("final"):
                outcome["final"] = delta["final"]
            if delta.get("draft"):
                outcome["draft"] = delta["draft"]
            if delta.get("last_prompt"):
                outcome["prompt"] = delta["last_prompt"]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _safe_stream(gen):
    """처리 중 오류가 나면 연결만 끊기지 않게, 화면에 Error 이벤트로 알리고 정상 종료한다.
    (승인 대기 중이던 요청은 그대로 남아 있어 다시 승인/거부할 수 있다.)"""
    try:
        async for chunk in gen:
            yield chunk
    except Exception as e:   # noqa: BLE001 — 사용자에게는 짧은 안내, 자세한 내용은 로그로
        log_event("ChatStreamError", error=f"{type(e).__name__}: {e}")
        yield _sse({"type": "Error", "message": f"처리 중 오류가 발생했습니다({type(e).__name__}). "
                                                "잠시 후 다시 시도해 주세요."})
        yield "data: [DONE]\n\n"


def _sse_response(gen) -> StreamingResponse:
    return StreamingResponse(_safe_stream(gen), media_type="text/event-stream", headers=SSE_HEADERS)


async def _chat_stream_gen(message: str, thread_id: str):
    """Wave 3 — 그래프의 노드별 진행 상황을 SSE로 스트리밍.

    이벤트마다 'data: {json}\\n\\n' 형태로 내려보내고 마지막에 data: [DONE] 센티널을 보낸다.
    [개선 I-9] 첫 이벤트로 Session(thread_id)을 보내 화면이 대화방 ID를 알게 한다.
    [After] 승인이 필요하면 ApprovalRequired 이벤트를 보내고 스트림을 끝낸다 → 화면이 승인/거부 버튼을
    보여주고, 버튼을 누르면 /chat/resume/stream으로 이어서 받는다."""
    deps = app.state.deps
    started = time.perf_counter()
    yield _sse({"type": "Session", "thread_id": thread_id})
    if pend := deps.session_store.get_pending(thread_id):
        yield _sse({"type": "Error", **_pending_notice(thread_id, pend)})
        yield "data: [DONE]\n\n"
        return
    history = deps.session_store.load(thread_id)
    if deps.version != "v2":
        result = await run_v1(deps, message, thread_id, history)
        for ev in result["events"]:
            yield _sse(ev)
        yield _sse({"type": "FinalAnswer", "reply": result["final"]})
    else:
        run_id = uuid.uuid4().hex
        outcome: dict = {}
        async for chunk in _stream_run(_new_input(message, thread_id, history),
                                       _graph_config(run_id), 0, outcome, thread_id):
            yield chunk
        if outcome.get("approval"):
            deps.session_store.set_pending(thread_id, run_id, message)
            yield _sse({**outcome["approval"], "thread_id": thread_id, "draft": outcome.get("draft", "")})
        else:
            yield _sse({"type": "FinalAnswer", "reply": outcome.get("final", ""),
                        "prompt": outcome.get("prompt")})
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", outcome.get("final", ""))
            await forget_run(app.state.checkpointer, run_id)
        log_event("ChatStreamFinished", thread_id=thread_id, run_id=run_id,
                  approval_required=bool(outcome.get("approval")),
                  elapsed_ms=round((time.perf_counter() - started) * 1000))
    yield "data: [DONE]\n\n"


def _resume_value(decision: str, comment: str, actions: list[dict] | None) -> dict:
    return {"decision": decision, "comment": comment, "actions": actions}


async def _resume_stream_gen(thread_id: str, decision: str, comment: str, actions: list[dict] | None = None):
    """[After 추가] 승인/거부 결정을 받아 멈췄던 그래프를 이어서 실행하며 SSE로 보낸다."""
    deps = app.state.deps
    pend = deps.session_store.get_pending(thread_id)
    if not pend:
        yield _sse({"type": "Error", "message": "승인 대기 중인 요청이 없습니다."})
        yield "data: [DONE]\n\n"
        return
    config = _graph_config(pend["run_id"])
    snapshot = await app.state.graph.aget_state(config)
    seen = len(snapshot.values.get("events", []))
    outcome: dict = {}
    async for chunk in _stream_run(Command(resume=_resume_value(decision, comment, actions)),
                                   config, seen, outcome, thread_id):
        yield chunk
    if outcome.get("approval"):   # 수정 요청 → 운영자 판단을 반영해 다시 쓴 보고서로 또 승인 대기(같은 실행 이어감)
        yield _sse({**outcome["approval"], "thread_id": thread_id, "draft": outcome.get("draft", "")})
        yield "data: [DONE]\n\n"
        return
    deps.session_store.pop_pending(thread_id)
    await forget_run(app.state.checkpointer, pend["run_id"])
    yield _sse({"type": "FinalAnswer", "reply": outcome.get("final", ""), "prompt": outcome.get("prompt")})
    deps.session_store.append(thread_id, "user", pend["message"])
    deps.session_store.append(thread_id, "assistant", outcome.get("final", ""))
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream_post(req: ChatRequest):
    """[개선 I-7] 강의 방식 — 질문을 본문(JSON)으로 받는다. 화면(web_chat.html)은 이 엔드포인트를 쓴다."""
    return _sse_response(_chat_stream_gen(req.message, _new_thread_id(req.thread_id)))


@app.get("/chat/stream")
async def chat_stream(message: str, thread_id: str | None = None):
    """curl -N 확인용 GET 버전(질문이 URL에 남으므로 화면에서는 POST를 쓴다)."""
    return _sse_response(_chat_stream_gen(message, _new_thread_id(thread_id)))


@app.post("/chat/resume/stream")
async def chat_resume_stream_post(req: ResumeRequest):
    return _sse_response(_resume_stream_gen(req.thread_id, req.decision, req.comment, req.actions))


@app.get("/chat/resume/stream")
async def chat_resume_stream(thread_id: str, decision: str = "approve", comment: str = ""):
    return _sse_response(_resume_stream_gen(thread_id, decision, comment))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
