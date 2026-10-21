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
"""
import asyncio
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from .config import RULES, incident_types_by_priority, types_from_alerts
from .deps import SupervisorDeps
from .equipment import stage_of
from .retry import with_retry

TH = RULES["thresholds"]


def specialist_for(equipment_id: str) -> str:
    stage = stage_of(equipment_id)
    for key, spec in RULES["specialists"].items():
        if stage in spec["stages"]:
            return key
    return "storage"


def _pct(a: float, b: float) -> float:
    return round(a / b * 100, 1) if b else 0.0


def stage_checks(status: dict) -> list[tuple[str, str | None]]:
    """설비 1대의 계측값을 기준값과 비교한다. (점검 문장, 의심 이상 유형) 목록을 돌려준다."""
    eq, m = status["equipment_id"], status.get("metrics", {})
    out = []
    if "pr_percent" in m and m["pr_percent"] < TH["pv_pr_min_percent"]:
        out.append((f"{eq} 발전 효율 {m['pr_percent']}% < 기준 {TH['pv_pr_min_percent']}%", "pv-underperform"))
    if "today_peak_kw" in m and m.get("contract_kw"):
        ratio = _pct(m["today_peak_kw"], m["contract_kw"])
        if ratio >= TH["peak_warn_ratio_percent"]:
            out.append((f"{eq} 오늘 최대수요 {m['today_peak_kw']}kW = 계약전력 {m['contract_kw']}kW의 {ratio}%",
                        "peak-demand"))
    if m.get("export_kw"):
        out.append((f"{eq} 역송 {m['export_kw']}kW 발생", None))
    if "bus_voltage_v" in m and m.get("rated_voltage_v"):
        rise = round(_pct(m["bus_voltage_v"], m["rated_voltage_v"]) - 100, 1)
        if rise > TH["voltage_rise_max_percent"]:
            out.append((f"{eq} 모선 전압 {m['bus_voltage_v']}V = 정격 {m['rated_voltage_v']}V 대비 +{rise}%",
                        "voltage-rise"))
    if "soc_percent" in m:
        if m["soc_percent"] > m.get("soc_upper_limit", 100):
            out.append((f"{eq} SOC {m['soc_percent']}% > 충전 상한 {m['soc_upper_limit']}%", "ess-soc-limit"))
        if m["soc_percent"] < m.get("soc_lower_limit", 0):
            out.append((f"{eq} SOC {m['soc_percent']}% < 방전 하한 {m['soc_lower_limit']}%", "ess-soc-low"))
    if m.get("cell_temp_max_c", 0) >= TH["ess_cell_temp_alarm_c"]:
        out.append((f"{eq} 셀 최고온도 {m['cell_temp_max_c']}°C ≥ 경보 {TH['ess_cell_temp_alarm_c']}°C",
                    "ess-overheat"))
    for key in ("data_age_min", "last_data_min_ago"):
        if m.get(key, 0) >= TH["data_stale_min"]:
            out.append((f"{eq} 데이터 {m[key]}분 경과 ≥ 기준 {TH['data_stale_min']}분", "comm-loss"))
    if m.get("missing_rate_percent", 0) > TH["missing_rate_max_percent"]:
        out.append((f"{eq} 결측률 {m['missing_rate_percent']}% > 기준 {TH['missing_rate_max_percent']}%",
                    "comm-loss"))
    return out


class SpecialistState(TypedDict, total=False):
    key: str
    name: str
    targets: list[str]
    collected: list[dict]
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
            data, retries = await with_retry(deps.mcp.call_tool, "get_equipment_status",
                                             {"equipment_id": eq_id})
            return eq_id, data, retries
        results = await asyncio.gather(*(one(t) for t in state["targets"]))
        return {"collected": [{"source": "monitoring", "data": d} for _, d, _ in results],
                "retries": sum(r for _, _, r in results),
                "task_events": [{"type": "TaskCompleted", "source": "monitoring", "target": eq,
                                 "retries": r, "agent": state["name"]} for eq, _, r in results]}

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
        if state["key"] == "grid":   # 계통 에이전트만 전력 수지 Tool을 쓴다
            balance = await deps.mcp.call_tool("get_power_balance", {})
            checks.append(f"전력 수지: 공급 {balance['supply_kw']}kW / 수요 {balance['demand_kw']}kW "
                          f"(차이 {balance['imbalance_kw']}kW)")
            if abs(balance["imbalance_kw"]) > TH["power_balance_tolerance_kw"]:
                suspected.append("comm-loss")
        order = incident_types_by_priority()
        suspected = sorted(set(suspected), key=order.index)
        summary, usage = await asyncio.to_thread(deps.fast_llm.analyze, state["collected"])
        return {"checks": checks, "suspected": suspected, "summary": summary.strip(),
                "power_balance": balance, "cost_usd": usage["usd"]}

    g = StateGraph(SpecialistState)
    g.add_node("collect", collect)
    g.add_node("assess", assess)
    g.set_entry_point("collect")
    g.add_edge("collect", "assess")
    g.add_edge("assess", END)
    return g.compile()
