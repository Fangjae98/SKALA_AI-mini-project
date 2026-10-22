"""질문 3분류 — 잡담 / 설비 조회 / 매뉴얼 조회(이상 대응).

인사·도움말·매뉴얼 목록·설비 조회는 AI 없이 코드가 데이터로 답하고, 증상이 있으면 항상 이상 대응으로 간다."""
import asyncio
import uuid

import pytest

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.intent import classify_by_rules, quick_reply_kind
from agent.mcp_client import MCPClient
from agent.session_store import SqliteSessionStore


@pytest.mark.parametrize("query, expected", [
    ("안녕", "chat"),
    ("안녕, 너는 뭘 할 수 있어?", "chat"),
    ("그럼 메뉴얼을 통해 확인 가능한 내용은 어떤게 있는지 설명해줘", "chat"),
    ("현재 내가 확인 가능한 설비들 종류 알려줘", "equipment"),
    ("ESS 1호기 상태 알려줘", "equipment"),
    ("변압기 상태 어때?", "equipment"),
    ("ESS 1호기 온도가 계속 올라가요", "incident"),
    ("옥외 ESS 잔량이 바닥이에요", "incident"),          # 작은 AI 모델이 설비 조회로 착각하던 질문
    ("ESS 1호기 상태 알려줘, 온도가 계속 올라가요", "incident"),   # 증상이 섞이면 이상 대응
    ("오늘 날씨 어때?", None),                             # 규칙에 없음 → AI가 분류
])
def test_질문_3분류_규칙(query, expected):
    assert classify_by_rules(query) == expected


def test_빠른_답변_종류():
    assert quick_reply_kind("안녕") == "greeting"
    assert quick_reply_kind("너는 뭘 할 수 있어?") == "help"
    assert quick_reply_kind("어떤 매뉴얼이 있어?") == "manual_list"
    assert quick_reply_kind("안녕 ESS 온도가 올라가요") is None


async def _ask(query, tmp_path):
    mcp = MCPClient(flaky_first_tool_call=False)
    await mcp.start()
    try:
        g = build_graph(SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp,
                                       SqliteSessionStore(tmp_path / "s.db"), "v2"))
        return await g.ainvoke({"query": query, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0},
                               {"configurable": {"thread_id": uuid.uuid4().hex}})
    finally:
        await mcp.stop()


def test_설비_목록은_데이터로_답한다(tmp_path):
    r = asyncio.run(_ask("현재 내가 확인 가능한 설비들 종류 알려줘", tmp_path))
    assert r["route"] == "equipment"
    assert "9대" in r["final"] and "ESS 1호기(옥내)(ess-01): 이상" in r["final"]
    assert "단지 부하(load-01): 정상" in r["final"]


def test_특정_설비_상태는_계측값을_그대로(tmp_path):
    r = asyncio.run(_ask("ESS 1호기 상태 알려줘", tmp_path))
    assert "충전율(SOC) 86%" in r["final"] and "셀 최고온도 46°C" in r["final"]
    assert not r.get("__interrupt__")                 # 조회만 — 승인 단계로 가지 않음


def test_매뉴얼_목록은_파일에서(tmp_path):
    r = asyncio.run(_ask("어떤 매뉴얼이 있어?", tmp_path))
    assert "대응 매뉴얼 9종" in r["final"] and "ESS 셀 과열" in r["final"]
