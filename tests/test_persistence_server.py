"""[개선 I-3] 재시작 후 승인 재개, [개선 I-9·I-7] API 동작, [개선 I-2] 블로킹 제거, [개선 I-11] MCP 동시 호출 테스트."""
import asyncio
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.persistence import create_checkpointer
from agent.session_store import SqliteSessionStore

INPUT = {"query": "ESS 1호기 온도가 계속 올라가요", "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0}


# ── I-3: 서버 재시작 후에도 승인 대기를 이어서 처리 ─────────────────────────
def test_재시작해도_승인_대기를_이어서_처리한다(tmp_path):
    db = tmp_path / "checkpoints.db"
    run_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": run_id}}

    async def first_process():
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            async with create_checkpointer("sqlite", db) as cp:
                deps = SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp, SqliteSessionStore(tmp_path / "s.db"), "v2")
                return await build_graph(deps, cp).ainvoke(INPUT, cfg)
        finally:
            await mcp.stop()

    async def second_process():   # 완전히 새 MCP·그래프·체크포인터 — 같은 파일만 공유
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            async with create_checkpointer("sqlite", db) as cp:
                deps = SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp, SqliteSessionStore(tmp_path / "s.db"), "v2")
                return await build_graph(deps, cp).ainvoke(Command(resume={"decision": "reject"}), cfg)
        finally:
            await mcp.stop()

    first = asyncio.run(first_process())
    assert "__interrupt__" in first
    second = asyncio.run(second_process())
    assert "[운영자 거부]" in second["final"]


# ── I-9·I-7: API ────────────────────────────────────────────────────────
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_DB", str(tmp_path / "session.db"))
    import server
    with TestClient(server.app) as c:
        yield c


def test_thread_id가_없으면_서버가_새로_만든다(client):
    a = client.post("/chat", json={"message": "안녕"}).json()
    b = client.post("/chat", json={"message": "안녕"}).json()
    assert a["thread_id"] != b["thread_id"] and a["thread_id"].startswith("t-")


def test_승인_대기_중_새_질문은_덮어쓰지_않는다(client):
    tid = "t-test-pending"
    first = client.post("/chat", json={"message": "ESS 1호기 온도가 계속 올라가요", "thread_id": tid}).json()
    assert first["status"] == "approval_required"
    second = client.post("/chat", json={"message": "DR 감축 요청이 왔어요", "thread_id": tid}).json()
    assert second["status"] == "approval_pending"
    done = client.post("/chat/resume", json={"thread_id": tid, "decision": "approve"}).json()
    assert done["status"] == "done" and "[운영자 승인 완료]" in done["reply"]


def test_POST_스트림과_버퍼링_방지_헤더(client):
    with client.stream("POST", "/chat/stream", json={"message": "안녕"}) as r:
        assert r.headers["x-accel-buffering"] == "no" and r.headers["cache-control"] == "no-cache"
        body = "".join(r.iter_text())
    events = [json.loads(line[5:]) for line in body.split("\n") if line.startswith("data:") and "[DONE]" not in line]
    assert events[0]["type"] == "Session" and events[-1]["type"] == "FinalAnswer"


# ── I-2: 실모델 호출이 이벤트 루프를 막지 않는다 ─────────────────────────────
class _SlowOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(0.5)   # 모델이 답하는 시간
        body = json.dumps({"message": {"content": '{"route": "chat", "needs_analysis": false}'},
                           "prompt_eval_count": 1, "eval_count": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_Ollama_호출_두개가_동시에_진행된다(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import agent.ollama_llm as ol
    monkeypatch.setattr(ol, "OLLAMA_URL", f"http://127.0.0.1:{srv.server_port}/api/chat")
    llm = ol.OllamaFastLLM("stub", 0)

    async def two():
        started = time.perf_counter()
        await asyncio.gather(llm.route("안녕"), llm.route("반가워"))
        return time.perf_counter() - started

    try:
        elapsed = asyncio.run(two())
    finally:
        srv.shutdown()
    assert elapsed < 0.9   # 막혔다면 0.5 + 0.5 = 1초 이상


# ── I-11: MCP 동시 요청이 섞이지 않는다 ─────────────────────────────────────
def test_MCP_동시_호출_결과가_섞이지_않는다():
    ids = ["gw-01", "pv-01", "pv-02", "pcc-01", "tr-01", "ess-01", "ess-02", "load-01", "evc-01"] * 3

    async def fn():
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            return await asyncio.gather(*(mcp.call_tool("get_equipment_status", {"equipment_id": i}) for i in ids))
        finally:
            await mcp.stop()

    results = asyncio.run(fn())
    assert [r["equipment_id"] for r in results] == ids


# ── 승인 단계에서 운영자가 고치기: 일부만 승인·문구 수정 / 판단을 적어 다시 작성 ──────────────────
def _approval_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_DB", str(tmp_path / "s.db"))
    import server
    return TestClient(server.app)


def test_일부만_골라서_문구를_고쳐_승인한다(tmp_path, monkeypatch):
    with _approval_client(tmp_path, monkeypatch) as client:
        first = client.post("/chat", json={"message": "ESS 1호기 온도가 계속 올라가요", "thread_id": "t-pick"}).json()
        acts = first["approval"]["actions"]
        assert len(acts) == 2
        picks = [{"original": acts[0], "text": "충방전 출력을 50%로 제한한다(운영자 승인 필요).", "checked": True},
                 {"original": acts[1], "text": acts[1], "checked": False}]
        done = client.post("/chat/resume", json={"thread_id": "t-pick", "decision": "approve", "actions": picks}).json()
        assert done["status"] == "done"
        assert "충방전 출력을 50%로 제한한다(운영자 승인 필요). (운영자 수정)" in done["reply"]
        assert "(운영자 제외 — 시행하지 않음)" in done["reply"]
        assert "1건을 승인했습니다(문구 수정 1건), 1건은 제외했습니다" in done["reply"]


def test_운영자_판단을_반영해_다시_쓰고_다시_승인받는다(tmp_path, monkeypatch):
    with _approval_client(tmp_path, monkeypatch) as client:
        client.post("/chat", json={"message": "ESS 1호기 온도가 계속 올라가요", "thread_id": "t-rev"})
        again = client.post("/chat/resume", json={"thread_id": "t-rev", "decision": "revise",
                                                  "comment": "공조기 점검이 끝날 때까지 출력은 50%로 유지"}).json()
        assert again["status"] == "approval_required"                     # 다시 쓴 보고서로 또 승인 대기
        assert "운영자 판단 반영: 공조기 점검이 끝날 때까지 출력은 50%로 유지" in again["draft"]
        assert any(e["type"] == "ApprovalRevise" for e in again["events"])
        done = client.post("/chat/resume", json={"thread_id": "t-rev", "decision": "approve"}).json()
        assert done["status"] == "done" and "[운영자 승인 완료]" in done["reply"]
