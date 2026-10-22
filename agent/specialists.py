"""[After 추가 — Multi-Agent] 단계별 전문 에이전트.

Before에서는 research 노드 하나가 모든 설비를 똑같이 조회만 하고, 판단은 알람 문구 키워드에 맡겼다.
After에서는 전력 흐름 단계마다 전문 에이전트를 두고, research 노드(Supervisor)가 이들을 동시에 부른다.

    research (Supervisor) ──┬── 데이터 수집 에이전트 (수집: gw)
                            ├── 발전 에이전트       (생산: pv)
                            ├── 계통 에이전트       (이동: pcc, tr) + 전력 수지 Tool
                            └── 저장·공급 에이전트  (저장·공급: ess, load, evc)

각 전문 에이전트는 작은 LangGraph(서브그래프)다:  collect(자기 담당 설비 계측 조회) → assess(점검)
- assess는 계측값을 config/rules.json의 기준값과 '코드로' 비교해 이상을 찾고(숫자 판단은 LLM에 맡기지 않음),
  fast LLM이 담당 단계의 상황을 한 문장으로 요약한다.
- 결과(점검 목록·의심 이상 유형·요약)를 Supervisor에게 돌려준다.

[개선 I-1 — 강의 12장 Error 격리] 설비 1대 조회가 실패해도(ToolError) 그 설비만 failed로 기록하고 나머지 설비는
계속 점검한다. 전력 수지 계산·LLM 요약이 실패해도 점검 결과는 그대로 돌려준다(부분 결과).
[개선 I-2] 요약은 asyncio.to_thread가 아니라 await로 부른다(LLM 백엔드가 모두 async가 됨).
"""
import asyncio
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from .config import RULES, incident_types_by_priority, types_from_alerts
from .deps import SupervisorDeps
from .diagnosis import judge
from .equipment import stage_of
from .retry import ToolError, with_retry

TH = RULES["thresholds"]


def specialist_for(equipment_id: str) -> str:
    stage = stage_of(equipment_id)
    for key, spec in RULES["specialists"].items():
        if stage in spec["stages"]:
            return key
    return "storage"


def stage_checks(status: dict) -> list[tuple[str, str | None]]:
    """설비 1대의 계측값을 설비별 상태 판단 기준(status_criteria)과 비교한다. (점검 문장, 의심 이상 유형) 목록.
    판정 자체는 agent/diagnosis.py — 설비 조회 화면·설비 현황 패널과 같은 기준을 쓴다(기준이 한 곳)."""
    return [(f["check"], f["incident"]) for f in judge(status)
            if f["level"] in ("warn", "alarm", "info") and f["check"]]


class SpecialistState(TypedDict, total=False):
    key: str
    name: str
    targets: list[str]
    collected: list[dict]
    failures: list[dict]
    retries: int
    task_events: list[dict]
    checks: list[str]
    suspected: list[str]
    summary: str
    power_balance: dict | None
    cost_usd: float


def build_specialist_graph(deps: SupervisorDeps):
    @traceable(name="specialist.collect")
    async def collect(state: SpecialistState) -> dict:
        async def one(eq_id: str):
            await asyncio.sleep(0.35)   # Base와 같은 태스크당 고정 지연 — 병렬 효과를 실측하기 위함
            try:
                data, retries = await with_retry(deps.mcp.call_tool, "get_equipment_status",
                                                 {"equipment_id": eq_id})
                return eq_id, data, retries, None
            except ToolError as e:   # 이 설비만 실패 — 나머지 설비 조회는 계속
                return eq_id, None, e.retries, e.to_dict()
        results = await asyncio.gather(*(one(t) for t in state["targets"]))
        events = []
        for eq, data, r, err in results:
            if err:
                events.append({"type": "TaskFailed", "source": "monitoring", "target": eq, "retries": r,
                               "agent": state["name"], "code": err["code"], "message": err["message"]})
            else:
                events.append({"type": "TaskCompleted", "source": "monitoring", "target": eq,
                               "retries": r, "agent": state["name"]})
        return {"collected": [{"source": "monitoring", "data": d} for _, d, _, err in results if not err],
                "failures": [{"target": eq, **err} for eq, _, _, err in results if err],
                "retries": sum(r for _, _, r, _ in results),
                "task_events": events}

    @traceable(name="specialist.assess")
    async def assess(state: SpecialistState) -> dict:
        checks, suspected, alerts = [], [], []
        for c in state["collected"]:
            for text, itype in stage_checks(c["data"]):
                checks.append(text)
                if itype:
                    suspected.append(itype)
            alerts.extend(c["data"].get("alerts", []))
        suspected.extend(types_from_alerts(alerts))   # 알람 문구로만 알 수 있는 이상(출력제어·DR 등)
        balance = None
        for f in state.get("failures", []):
            # 계측값을 못 받았다는 것 자체가 데이터 수집 경로 이상 신호 → 통신 두절 의심(매뉴얼 선택·게이트웨이 재조사에 반영)
            checks.append(f"{f['target']} 조회 실패({f['code']}) — 판단에서 제외, 데이터 수집 경로 이상 의심")
            suspected.append("comm-loss")
        if state["key"] == "grid":   # 계통 에이전트만 전력 수지 Tool을 쓴다
            try:
                balance, _ = await with_retry(deps.mcp.call_tool, "get_power_balance", {})
                checks.append(f"전력 수지: 공급 {balance['supply_kw']}kW / 수요 {balance['demand_kw']}kW "
                              f"(차이 {balance['imbalance_kw']}kW)")
                if abs(balance["imbalance_kw"]) > TH["power_balance_tolerance_kw"]:
                    suspected.append("comm-loss")
            except ToolError as e:
                checks.append(f"전력 수지 계산 실패({e.code}) — 수지 점검 생략")
        order = incident_types_by_priority()
        suspected = sorted(set(suspected), key=order.index)
        cost = 0.0
        if state["collected"]:
            try:
                summary, usage = await deps.fast_llm.analyze(state["collected"])
                cost = usage["usd"]
            except Exception as e:   # 요약은 부가 정보 — 실패해도 숫자 점검 결과는 그대로 쓴다
                summary = f"(상황 요약 실패: {type(e).__name__})"
        else:
            summary = "담당 설비 데이터를 하나도 받지 못함"
        return {"checks": checks, "suspected": suspected, "summary": summary.strip(),
                "power_balance": balance, "cost_usd": cost}

    g = StateGraph(SpecialistState)
    g.add_node("collect", collect)
    g.add_node("assess", assess)
    g.set_entry_point("collect")
    g.add_edge("collect", "assess")
    g.add_edge("assess", END)
    return g.compile()
