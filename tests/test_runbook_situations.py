"""상황별 매뉴얼 — 매뉴얼 하나에 '## 상황 N: …' 절을 여러 개 두고, 질문에 맞는 상황의 조치를 쓴다.

예시 매뉴얼: data/runbooks/ess-overheat.md (기본 절차 + 상황 6개)."""
import asyncio
import uuid

import pytest

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.rag import RUNBOOK_DIR, RunbookIndex, TfidfEmbedder, runbook_excerpt, split_runbook
from agent.session_store import SqliteSessionStore

OVERHEAT = (RUNBOOK_DIR / "ess-overheat.md").read_text()
FIRE = "연기·이상 냄새·가스 감지 등 화재 징후"
HVAC = "공조(HVAC)·냉각 장치 고장으로 ESS실 온도 상승"


def test_매뉴얼을_기본_부분과_상황_절로_나눈다():
    base, situations = split_runbook(OVERHEAT)
    assert "## 기본 절차" in base and "## 상황" not in base
    assert situations[0]["title"] == FIRE                      # 위험한 상황이 맨 위
    assert len(situations) == 6


def test_별도_절차_상황은_기본_절차를_빼고_그_상황만():
    ex = runbook_excerpt(OVERHEAT, FIRE)
    assert "대피" in ex and "119" in ex
    assert "get_equipment_status로 셀 최고온도" not in ex      # 기본 절차 대신 이 상황 절차만


def test_기본_절차_추가_상황은_둘_다():
    ex = runbook_excerpt(OVERHEAT, HVAC)
    assert "충방전을 즉시 중지한다" in ex and "예비 공조기" in ex


def test_상황을_못_찾으면_기본_부분만():
    ex = runbook_excerpt(OVERHEAT, None)
    assert "## 기본 절차" in ex and "## 상황" not in ex and "## 해당 상황" not in ex


@pytest.mark.parametrize("query, situation", [
    ("ESS실에서 타는 냄새가 나요", FIRE),
    ("타는 냄새가 나고 에어컨도 꺼졌어요", FIRE),             # 키워드가 겹치면 위쪽(더 위험한) 상황
    ("ESS실 에어컨이 꺼져서 온도가 올라가요", HVAC),
    ("ESS 1호기 한 랙만 온도가 높아요", "특정 랙·모듈만 온도가 높음 (온도 편차)"),
    ("ESS 온도 센서 값이 갑자기 튀어요", "온도 센서 오류가 의심됨 (값이 갑자기 튐)"),
])
def test_질문에_맞는_상황을_고른다(query, situation):
    hits = RunbookIndex(TfidfEmbedder()).search(query, 3)
    top = next(h for h in hits if h["incident_type"] == "ess-overheat")
    assert top["situation"] == situation


def test_상황_절이_다른_매뉴얼_선택을_흐리지_않는다():
    """'EV 충전기' 질문이 과열 매뉴얼의 '충전 중' 절에 끌려가지 않는다(매뉴얼 선택은 기본 부분 점수로)."""
    hits = RunbookIndex(TfidfEmbedder()).search("EV 충전기 데이터가 안 들어오는 것 같아요", 3)
    assert hits[0]["incident_type"] != "ess-overheat"


def test_화재_상황이면_대피부터_보고서_맨_앞에(tmp_path):
    async def run():
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            g = build_graph(SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp,
                                           SqliteSessionStore(tmp_path / "s.db"), "v2"))
            return await g.ainvoke({"query": "ESS 1호기 쪽 ESS실에서 타는 냄새가 나요", "thread_id": "t",
                                    "history": [], "events": [], "cost_usd": 0.0},
                                   {"configurable": {"thread_id": uuid.uuid4().hex}})
        finally:
            await mcp.stop()
    r = asyncio.run(run())
    draft = r["draft"]
    assert "대피" in draft and draft.index("대피") < draft.index("충전을 중지")   # 화재 대응이 충전 상한보다 먼저
    assert any(e.get("situation") == FIRE for e in r["events"] if e["type"] == "ReplanCheck")


# ── 9개 매뉴얼 전체 — 매뉴얼마다 대표 상황 하나씩, 전체 흐름(데이터 근거 포함)으로 확인 ──────────
ALL_MANUAL_CASES = [
    ("게이트웨이 전원이 나가서 모든 설비 값이 전부 안 들어와요", "comm-loss", "게이트웨이 전체"),
    ("출력제어 지시가 왔는데 ESS가 가득 차 있어요", "curtailment", "ESS가 가득"),
    ("DR 감축이 이미 시작됐는데 지금 바로 줄여야 해요", "dr-event", "이미 감축"),
    ("ESS 1호기 충전 상한 설정값이 잘못된 것 같아요", "ess-soc-limit", "EMS·BMS"),
    ("정전이라 ESS로 버티는데 잔량이 바닥이에요", "ess-soc-low", "정전"),
    ("냉방이 한꺼번에 돌아서 피크가 계약전력에 닿았어요", "peak-demand", "냉난방"),
    ("2호기 스트링 하나가 0A예요", "pv-underperform", "특정 스트링"),
    ("밤에도 변압기 전압이 높아요", "voltage-rise", "밤에도"),
    ("ESS실에서 타는 냄새가 나요", "ess-overheat", "연기"),
]


def test_모든_매뉴얼에서_상황을_고른다(tmp_path):
    async def run():
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            g = build_graph(SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp,
                                           SqliteSessionStore(tmp_path / "s.db"), "v2"))
            out = []
            for q, _, _ in ALL_MANUAL_CASES:
                r = await g.ainvoke({"query": q, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0},
                                    {"configurable": {"thread_id": uuid.uuid4().hex}})
                ev = next(e for e in r["events"] if e["type"] == "ReplanCheck" and not e["is_sufficient"])
                out.append((ev["target"], ev.get("situation") or ""))
            return out
        finally:
            await mcp.stop()
    got = asyncio.run(run())
    for (q, manual, situation), (target, sit) in zip(ALL_MANUAL_CASES, got):
        assert target == manual and sit.startswith(situation), f"{q} → {target} / {sit}"
