"""그래프 전체 흐름 테스트 — fake 두뇌 + 실제 MCP 서버(별도 프로세스)로 끝까지 실행한다."""
import asyncio
import uuid

from langgraph.types import Command

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.session_store import SqliteSessionStore


def run(coro):
    return asyncio.run(coro)


async def _with_graph(fn, tmp_path):
    mcp = MCPClient()
    await mcp.start()
    try:
        deps = SupervisorDeps(FastLLM("fast", 0.00015), QualityLLM("quality", 0.005), mcp,
                              SqliteSessionStore(tmp_path / "s.db"), "v2")
        return await fn(build_graph(deps))
    finally:
        await mcp.stop()


def _inp(q):
    return {"query": q, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0}


def test_잡담은_바로_답한다(tmp_path):
    async def fn(g):
        return await g.ainvoke(_inp("안녕, 너는 뭘 할 수 있어?"), {"configurable": {"thread_id": uuid.uuid4().hex}})
    r = run(_with_graph(fn, tmp_path))
    assert r["route"] == "chat" and r["final"]


def test_제어조치는_승인을_기다렸다가_이어진다(tmp_path):
    async def fn(g):
        cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
        first = await g.ainvoke(_inp("ESS 1호기 온도가 계속 올라가요"), cfg)
        second = await g.ainvoke(Command(resume={"decision": "reject"}), cfg)
        return first, second
    first, second = run(_with_graph(fn, tmp_path))
    assert "__interrupt__" in first                                   # 승인 대기에서 멈춤
    assert first["__interrupt__"][0].value["actions"]
    replan = next(e for e in first["events"] if e["type"] == "ReplanCheck")
    assert replan["target"] == "ess-overheat" and replan["method"] == "rag"
    assert "[운영자 거부]" in second["final"]                         # 거부 결과가 최종 답에 반영
