"""분산에너지 운영 에이전트 — MCP Server ("ops").

Tool / Resource / Prompt 3대 capability를 모두 노출하는 fastmcp 서버.
FastAPI 백엔드(server.py)가 lifespan에서 이 파일을 subprocess(stdio)로 1회 기동해
프로세스가 살아 있는 동안 재사용한다.

- Tool     get_equipment_status : 설비 1대의 현재 계측값과 알람 조회
- Resource runbook://der-response/{incident_type} : 이상 유형별 대응 매뉴얼(정적 문서)
- Prompt   incident_analysis_prompt : 분산에너지 이상 진단 표준 절차
- [After 추가] Tool search_runbooks  : 질문과 의미가 가까운 대응 매뉴얼 검색(RAG)
- [After 추가] Tool get_power_balance : 단지 전체 전력 수지(공급 합 = 수요 합) 계산

[Base 대비 변경] 서버 CPU/메모리 2칸 고정 → 설비별 metrics 묶음, 서버 이름 → 설비 ID,
runbook URI runbook://incident-response/ → runbook://der-response/,
진단 절차를 생산·이동·저장·공급 + 수집 단계 기준으로 변경.
계측값은 전력 수지(공급 = 수요)가 맞도록 고정 스냅샷을 그대로 돌려준다(Base의 무작위
흔들기 제거 — 숫자가 흔들리면 전력 수지와 보고서 수치를 검증할 수 없기 때문).
"""
import json
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from agent.config import RULES, embedding_backend
from agent.rag import RunbookIndex

mcp = FastMCP(name="der-ops-agent-ops")

DATA_DIR = Path(__file__).parent / "data"


class EquipmentStatus(BaseModel):
    equipment_id: str = Field(description="설비 ID (예: pv-02)")
    name: str = Field(description="설비 이름 (예: 태양광 2호기)")
    stage: str = Field(description="전력 흐름 단계: 수집 / 생산 / 이동 / 저장 / 공급")
    status: str = Field(description="상태: healthy / degraded / down / unknown")
    metrics: dict[str, int | float] = Field(default_factory=dict, description="계측값 묶음")
    last_checked: str = Field(description="조회 시각 (ISO 8601)")
    alerts: list[str] = Field(default_factory=list, description="활성 알람 목록")


@mcp.tool()
def get_equipment_status(equipment_id: str) -> EquipmentStatus:
    """지정한 분산에너지 설비의 현재 계측값과 알람을 조회합니다."""
    equipment = json.loads((DATA_DIR / "monitoring.json").read_text())
    now = datetime.now().isoformat(timespec="seconds")
    if equipment_id not in equipment:
        return EquipmentStatus(
            equipment_id=equipment_id, name="(미등록)", stage="-", status="unknown",
            last_checked=now, alerts=[f"설비 '{equipment_id}'은(는) 등록되지 않은 설비입니다"],
        )
    data = equipment[equipment_id]
    return EquipmentStatus(
        equipment_id=equipment_id,
        name=data["name"],
        stage=data["stage"],
        status=data["status"],
        metrics=data.get("metrics", {}),
        last_checked=now,
        alerts=data.get("alerts", []),
    )


@mcp.resource("runbook://der-response/{incident_type}")
def get_incident_runbook(incident_type: str) -> str:
    """이상 유형별 대응 매뉴얼(정적 문서)을 반환합니다."""
    runbook_path = DATA_DIR / "runbooks" / f"{incident_type}.md"
    if not runbook_path.exists():
        return f"# Runbook Not Found\n\nNo runbook for incident type: {incident_type}"
    return runbook_path.read_text()


_RUNBOOK_INDEX: RunbookIndex | None = None   # 첫 검색 때 한 번만 만들고 재사용(매뉴얼 임베딩 비용 절약)


@mcp.tool()
def search_runbooks(query: str, top_k: int = 3) -> dict:
    """[After 추가 — RAG] 질문과 의미가 가장 가까운 대응 매뉴얼을 점수 순으로 돌려줍니다."""
    global _RUNBOOK_INDEX
    if _RUNBOOK_INDEX is None:
        _RUNBOOK_INDEX = RunbookIndex()
    return {"backend": embedding_backend(), "hits": _RUNBOOK_INDEX.search(query, top_k)}


@mcp.tool()
def get_power_balance() -> dict:
    """[After 추가] 단지 전체 전력 수지를 계산합니다. 공급(발전·방전·수전) 합과 수요(부하·충전·역송) 합이
    같아야 정상이며, 차이가 크면 계측 누락이나 오류를 의심합니다."""
    equipment = json.loads((DATA_DIR / "monitoring.json").read_text())
    pb = RULES["power_balance"]
    supply, demand = [], []
    for eq_id, data in equipment.items():
        for key, value in data.get("metrics", {}).items():
            if key in pb["supply_metrics"] and value:
                supply.append({"equipment_id": eq_id, "metric": key, "kw": value})
            elif key in pb["demand_metrics"] and value:
                demand.append({"equipment_id": eq_id, "metric": key, "kw": value})
    supply_kw = sum(x["kw"] for x in supply)
    demand_kw = sum(x["kw"] for x in demand)
    return {"supply_kw": supply_kw, "demand_kw": demand_kw, "imbalance_kw": supply_kw - demand_kw,
            "supply": supply, "demand": demand}


@mcp.prompt()
def incident_analysis_prompt(system_name: str, symptom: str) -> str:
    """분산에너지 이상 진단 표준 절차 — planning 노드가 계획을 세울 때 이 절차를 따른다."""
    return f"""다음 분산에너지 설비 이상에 대해 표준 진단 절차를 수행하세요.

## 이상 정보
- 설비: {system_name}
- 증상: {symptom}

## 진단 절차 (반드시 순서대로 수행)
1. get_equipment_status Tool로 관련 설비의 현재 계측값과 알람을 조회하세요.
2. 이상이 전력 흐름의 어느 단계(생산·이동·저장·공급, 데이터 수집)에서 났는지 정하세요.
3. 증상에 해당하는 대응 매뉴얼(runbook://der-response/...)을 확인하세요.
4. 위 정보를 종합해 가능한 원인과 권장 조치를 제시하세요. 설비를 움직이는 조치는 운영자 승인 필요로 표시하세요.
"""


if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except TypeError:   # 구버전 fastmcp 호환
        mcp.run()
