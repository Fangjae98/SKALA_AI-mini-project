"""분산에너지 운영 에이전트 — FastAPI 백엔드 (Base all_total_vue와 동일 구조, Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨
- [After] HITL: 설비 제어 조치가 있으면 승인 대기 → POST /chat/resume 또는 GET /chat/resume/stream으로 재개
"""
import json
import os
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
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.session_store import SqliteSessionStore
from agent.supervisor_v1 import run_v1

CREATION_COUNT = 0   # Wave 1 — 프로세스 내 Agent 번들 생성 횟수 (항상 1이어야 정상)


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str
    thread_id: str = "default"


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
                             usd_per_1k_tokens=0.00015)
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
    app.state.graph = build_graph(deps) if deps.version == "v2" else None
    app.state.instance_id = id(deps)
    app.state.pending = {}   # [After] 승인 대기 중인 실행: thread_id → {run_id, message}
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
            "quality_llm": app.state.deps.quality_llm.name}


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
        "cost_usd": result.get("cost_usd", 0.0),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


# ── [After 추가 — HITL] 승인 대기 관리 ─────────────────────────────────────
# 그래프 실행 1건마다 고유 run_id로 체크포인트를 따로 둔다(같은 대화방이라도 질문마다 상태가 섞이지 않게).
# 승인 대기 중인 실행은 대화방(thread_id)별로 app.state.pending에 기억해 두고, 승인/거부가 오면 이어서 실행한다.
def _graph_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _new_input(message: str, thread_id: str, history: list[dict]) -> dict:
    return {"query": message, "thread_id": thread_id, "history": history, "events": [], "cost_usd": 0.0}


def _interrupt_value(result_or_payload) -> dict | None:
    intr = result_or_payload.get("__interrupt__") if isinstance(result_or_payload, dict) else None
    return intr[0].value if intr else None


class ResumeRequest(BaseModel):
    thread_id: str = "default"
    decision: str = "approve"   # approve | reject
    comment: str = ""


@traceable(name="chat_request")
async def _handle_chat(message: str, thread_id: str) -> tuple[dict, str | None]:
    deps = app.state.deps
    history = deps.session_store.load(thread_id)   # Wave 12
    if deps.version == "v2":
        run_id = uuid.uuid4().hex
        result = await app.state.graph.ainvoke(_new_input(message, thread_id, history), _graph_config(run_id))
        approval = _interrupt_value(result)
        if approval:   # 승인 대기 — 대화 기록은 승인/거부 후 최종 답변과 함께 저장
            app.state.pending[thread_id] = {"run_id": run_id, "message": message}
            result = {**result, "approval_request": approval}
            rt = get_current_run_tree()
            return result, (str(rt.id) if rt else None)
    else:
        result = await run_v1(deps, message, thread_id, history)   # Wave 14
    deps.session_store.append(thread_id, "user", message)
    deps.session_store.append(thread_id, "assistant", result["final"])
    rt = get_current_run_tree()
    return result, (str(rt.id) if rt else None)


@app.post("/chat")
async def chat(req: ChatRequest):
    result, run_id = await _handle_chat(req.message, req.thread_id)
    if result.get("approval_request"):
        return {"status": "approval_required", "draft": result.get("draft", ""),
                "approval": result["approval_request"], "events": result.get("events", []),
                "metadata": _metadata(result, run_id),
                "next": "POST /chat/resume {thread_id, decision: approve|reject}"}
    return {"status": "done", "reply": result["final"], "events": result.get("events", []),
            "metadata": _metadata(result, run_id)}


@app.post("/chat/resume")
async def chat_resume(req: ResumeRequest):
    """[After 추가] 승인 대기 중인 실행을 승인(approve) 또는 거부(reject)로 이어서 끝낸다."""
    pend = app.state.pending.pop(req.thread_id, None)
    if not pend:
        return {"status": "error", "message": f"thread_id={req.thread_id}에 승인 대기 중인 요청이 없습니다."}
    deps = app.state.deps
    result = await app.state.graph.ainvoke(
        Command(resume={"decision": req.decision, "comment": req.comment}), _graph_config(pend["run_id"]))
    deps.session_store.append(req.thread_id, "user", pend["message"])
    deps.session_store.append(req.thread_id, "assistant", result["final"])
    return {"status": "done", "reply": result["final"], "events": result.get("events", []),
            "metadata": _metadata(result, None)}


async def _stream_run(graph_input, config: dict, seen: int, outcome: dict):
    """그래프를 실행하며 SSE 문자열을 하나씩 내보낸다. 끝나면 outcome에 final / approval을 채운다.

    stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는 노드가 끝날 때마다의 상태 델타,
    "custom"은 write_node/chat_answer_node가 get_stream_writer()로 실행 중에 흘려보내는 토큰(AnswerToken)이다.
    승인 대기(interrupt)에 걸리면 updates에 '__interrupt__'가 들어온다."""
    async for mode, payload in app.state.graph.astream(graph_input, config,
                                                       stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            continue
        approval = _interrupt_value(payload)
        if approval:
            outcome["approval"] = approval
            continue
        for node_name, delta in payload.items():
            if not isinstance(delta, dict):
                continue
            for ev in (delta.get("events") or [])[seen:]:
                yield f"data: {json.dumps({**ev, 'node': node_name}, ensure_ascii=False)}\n\n"
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


@app.get("/chat/stream")
async def chat_stream(message: str, thread_id: str = "default"):
    """Wave 3 — 그래프의 노드별 진행 상황을 SSE로 스트리밍.

    이벤트마다 'data: {json}\\n\\n' 형태로 내려보내고 마지막에 data: [DONE] 센티널을 보낸다.
    [After] 승인이 필요하면 ApprovalRequired 이벤트를 보내고 스트림을 끝낸다 → 화면이 승인/거부 버튼을
    보여주고, 버튼을 누르면 /chat/resume/stream으로 이어서 받는다."""
    deps = app.state.deps

    async def gen():
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
                                           _graph_config(run_id), 0, outcome):
                yield chunk
            if outcome.get("approval"):
                app.state.pending[thread_id] = {"run_id": run_id, "message": message}
                yield _sse({**outcome["approval"], "thread_id": thread_id, "draft": outcome.get("draft", "")})
            else:
                yield _sse({"type": "FinalAnswer", "reply": outcome.get("final", ""),
                            "prompt": outcome.get("prompt")})
                deps.session_store.append(thread_id, "user", message)
                deps.session_store.append(thread_id, "assistant", outcome.get("final", ""))
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/chat/resume/stream")
async def chat_resume_stream(thread_id: str = "default", decision: str = "approve", comment: str = ""):
    """[After 추가] 승인/거부 결정을 받아 멈췄던 그래프를 이어서 실행하며 SSE로 보낸다."""
    deps = app.state.deps

    async def gen():
        pend = app.state.pending.pop(thread_id, None)
        if not pend:
            yield _sse({"type": "Error", "message": "승인 대기 중인 요청이 없습니다."})
            yield "data: [DONE]\n\n"
            return
        config = _graph_config(pend["run_id"])
        snapshot = await app.state.graph.aget_state(config)
        seen = len(snapshot.values.get("events", []))
        outcome: dict = {}
        async for chunk in _stream_run(Command(resume={"decision": decision, "comment": comment}),
                                       config, seen, outcome):
            yield chunk
        yield _sse({"type": "FinalAnswer", "reply": outcome.get("final", ""), "prompt": outcome.get("prompt")})
        deps.session_store.append(thread_id, "user", pend["message"])
        deps.session_store.append(thread_id, "assistant", outcome.get("final", ""))
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
