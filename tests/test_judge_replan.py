"""[개선 I-4] 코드 검사 + LLM 심판 혼합 검수, [개선 I-5] 연관 설비 재조사 테스트."""
import asyncio
import uuid

from langgraph.types import Command

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM, parse_judgement
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.session_store import SqliteSessionStore

RUNBOOK_OVERHEAT = "# Runbook: ESS 셀 과열(Overheat) 대응 절차\n\n1. 확인한다.\n2. 충방전을 중지한다.\n"
RUNBOOK_LIMIT = "# Runbook: ESS 충전 상한 초과(SOC Limit) 대응 절차\n\n1. 확인한다.\n2. 충전을 중지한다.\n"
COLLECTED = [
    {"source": "monitoring", "data": {"equipment_id": "ess-01"}},
    {"source": "runbook", "data": RUNBOOK_OVERHEAT, "role": "primary"},
    {"source": "runbook", "data": RUNBOOK_LIMIT, "role": "safety"},
]


def judge(draft):
    return asyncio.run(FastLLM("f", 0).judge("질문", COLLECTED, draft))[0]


def test_안전이슈가_빠지면_심판이_반려한다():
    j = judge("1) ess-01 과열 2) 'ESS 셀 과열(Overheat)' 조치")
    assert not j["is_acceptable"] and "ESS 충전 상한 초과(SOC Limit)" in j["reasoning"]


def test_안전이슈를_뒤에_다루면_도메인_기준_불합격():
    j = judge("1) ess-01 2) 'ESS 셀 과열(Overheat)' 3) 'ESS 충전 상한 초과(SOC Limit)'")
    assert not j["is_acceptable"] and j["reasoning"].startswith("Domain")


def test_모든_기준을_지키면_통과():
    assert judge("1) ess-01 2) 'ESS 충전 상한 초과(SOC Limit)' 3) 'ESS 셀 과열(Overheat)'")["is_acceptable"]


def test_심판_응답_형식이_깨지면_통과_처리():
    assert parse_judgement("not json")["is_acceptable"] is True


async def _ask(query, tmp_path):
    mcp = MCPClient(flaky_first_tool_call=False)
    await mcp.start()
    try:
        deps = SupervisorDeps(FastLLM("f", 0.00015), QualityLLM("q", 0.005), mcp,
                              SqliteSessionStore(tmp_path / "s.db"), "v2")
        g = build_graph(deps)
        cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
        r = await g.ainvoke({"query": query, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0}, cfg)
        while r.get("__interrupt__"):
            r = await g.ainvoke(Command(resume={"decision": "approve"}), cfg)
        return r
    finally:
        await mcp.stop()


def test_짧게_답하라는_요청에도_안전조치를_빼지_않는다(tmp_path):
    r = asyncio.run(_ask("ESS 1호기 온도가 계속 올라가요. 답변은 50자 이내로 아주 짧게 해주세요", tmp_path))
    verdicts = [e for e in r["events"] if e["type"] == "Validation"]
    assert any(v["judge"] and not v["judge"]["is_acceptable"] for v in verdicts)   # 짧은 초안을 심판이 반려
    assert "ESS 충전 상한 초과" in r["final"] and "ESS 셀 과열" in r["final"]


def test_통신두절이면_게이트웨이를_추가로_조사한다(tmp_path):
    r = asyncio.run(_ask("EV 충전기 데이터가 안 들어오는 것 같아요", tmp_path))
    replan = next(e for e in r["events"] if e["type"] == "ReplanCheck" and not e["is_sufficient"])
    assert replan["extra_targets"] == ["gw-01"]
    surveyed = [e["target"] for e in r["events"] if e["type"] == "TaskCompleted" and e["source"] == "monitoring"]
    assert "gw-01" in surveyed and "결측률" in r["final"]
