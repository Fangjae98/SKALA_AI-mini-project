"""[개선 I-1] 재시도 정책과 Error 격리 테스트 — 설비 하나가 실패해도 그래프가 멈추지 않고 부분 보고서를 만든다."""
import asyncio
import os
import uuid

import pytest
from langgraph.types import Command

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.retry import TOOL_EXECUTION_ERROR, TOOL_TIMEOUT_AFTER_RETRY, ToolError, with_retry
from agent.session_store import SqliteSessionStore


def run(coro):
    return asyncio.run(coro)


# ── 재시도 정책 ──────────────────────────────────────────────────────────
def test_일시오류는_재시도해서_복구한다():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise asyncio.TimeoutError()
        return "ok"

    result, retries = run(with_retry(flaky, base_delay=0.01))
    assert result == "ok" and retries == 2


def test_영구오류는_재시도하지_않고_표준오류로_바꾼다():
    calls = []

    async def broken():
        calls.append(1)
        raise ValueError("잘못된 인자")

    with pytest.raises(ToolError) as e:
        run(with_retry(broken, base_delay=0.01, tool_name="get_equipment_status"))
    assert e.value.code == TOOL_EXECUTION_ERROR and len(calls) == 1


def test_재시도를_다_써도_안되면_시간초과_오류():
    async def always_timeout():
        raise asyncio.TimeoutError()

    with pytest.raises(ToolError) as e:
        run(with_retry(always_timeout, base_delay=0.01))
    assert e.value.code == TOOL_TIMEOUT_AFTER_RETRY and e.value.retries == 2


# ── 그래프 전체: 장애 주입 ────────────────────────────────────────────────
class FailingMCP:
    """특정 설비 조회만 실패시키는 MCP 대역(나머지는 실제 MCP 서버로 전달)."""

    def __init__(self, inner, fail_targets=(), fail_runbooks=False):
        self._inner, self._fail, self._fail_rb = inner, set(fail_targets), fail_runbooks

    async def call_tool(self, name, arguments, server="ops"):
        if name == "get_equipment_status" and arguments.get("equipment_id") in self._fail:
            raise ConnectionError(f"{arguments['equipment_id']} 계측 서버 응답 없음")
        return await self._inner.call_tool(name, arguments, server)

    async def read_resource(self, uri, server):
        if self._fail_rb:
            raise ConnectionError("매뉴얼 저장소 응답 없음")
        return await self._inner.read_resource(uri, server)

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def _ask(query, tmp_path, **fail):
    mcp = MCPClient(flaky_first_tool_call=False)
    await mcp.start()
    try:
        deps = SupervisorDeps(FastLLM("fast", 0.00015), QualityLLM("quality", 0.005), FailingMCP(mcp, **fail),
                              SqliteSessionStore(tmp_path / "s.db"), "v2")
        g = build_graph(deps)
        cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
        r = await g.ainvoke({"query": query, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0}, cfg)
        while r.get("__interrupt__"):
            r = await g.ainvoke(Command(resume={"decision": "approve"}), cfg)
        return r
    finally:
        await mcp.stop()


def test_설비_하나가_실패해도_부분_보고서를_만든다(tmp_path):
    r = run(_ask("EV 충전기 데이터가 안 들어오는 것 같아요", tmp_path, fail_targets={"evc-01"}))
    assert [f["target"] for f in r["failures"]] == ["evc-01"]
    assert any(e["type"] == "TaskFailed" and e["target"] == "evc-01" for e in r["events"])
    assert "load-01" in r["final"]                         # 실패하지 않은 설비 정보는 그대로 들어감
    assert "※ 일부 데이터를 받지 못했습니다: evc-01" in r["final"]
    assert "확인 불가" in r["final"]                       # 작성 단계도 실패를 숨기지 않음
    replan = next(e for e in r["events"] if e["type"] == "ReplanCheck" and not e["is_sufficient"])
    assert replan["target"] == "comm-loss"                 # 조회 실패 = 통신 두절 의심 → '충전' 단어 착시에 끌려가지 않음
    assert replan["extra_targets"] == ["gw-01"]            # 게이트웨이까지 확인


def test_매뉴얼_조회가_실패해도_계측_결과로_답한다(tmp_path):
    r = run(_ask("ESS 1호기 온도가 계속 올라가요", tmp_path, fail_runbooks=True))
    assert r["final"] and any(f["target"].startswith("매뉴얼") for f in r["failures"])
    assert "셀 최고온도 46°C" in r["final"]


def test_MCP_서버의_장애주입_스위치(tmp_path, monkeypatch):
    """DER_FAULT_INJECT로 실제 MCP 서버(별도 프로세스)가 그 설비 조회만 실패시킨다."""
    monkeypatch.setenv("DER_FAULT_INJECT", "evc-01")

    async def fn():
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            deps = SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp, SqliteSessionStore(tmp_path / "s.db"), "v2")
            g = build_graph(deps)
            return await g.ainvoke({"query": "EV 충전기 데이터가 안 들어오는 것 같아요", "thread_id": "t",
                                    "history": [], "events": [], "cost_usd": 0.0},
                                   {"configurable": {"thread_id": uuid.uuid4().hex}})
        finally:
            await mcp.stop()

    r = run(fn())
    assert r["failures"][0]["target"] == "evc-01" and r["failures"][0]["code"] == TOOL_EXECUTION_ERROR
    assert os.environ["DER_FAULT_INJECT"] == "evc-01"
