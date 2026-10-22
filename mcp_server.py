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

[개선 I-8 — 강의 3장 주의점]
- Tool 3개에 ToolAnnotations(readOnlyHint=True: 조회만 함, openWorldHint=False: 단지 내부 데이터만 봄)를 붙이고,
  docstring에 입력·반환·제약을 적었다(docstring 품질이 LLM의 도구 호출 정확도를 좌우한다).
- 매뉴얼 Resource는 본문만이 아니라 last_modified·sha256을 함께 JSON으로 돌려준다. URI가 같아도 파일이
  바뀌었는지 알 수 있어야 캐시가 옛 매뉴얼을 조용히 돌려주는 사고를 막을 수 있다.
[개선 I-1 — 장애 주입] 환경변수 DER_FAULT_INJECT=gw-01,pv-02 처럼 설비 ID를 주면 그 설비 조회만 일부러 실패한다
(강의의 chmod 000 실습과 같은 목적: 설비 하나가 실패해도 나머지로 부분 보고서를 만드는지 확인).
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from agent.config import RULES, embedding_backend
from agent.diagnosis import generated_alerts, judge, overall_status
from agent.rag import RunbookIndex

mcp = FastMCP(name="der-ops-agent-ops")

DATA_DIR = Path(__file__).parent / "data"
MONITORING_FILE = Path(__file__).parent / os.environ.get("DER_MONITORING_FILE", "data/monitoring.json")


class EquipmentStatus(BaseModel):
    equipment_id: str = Field(description="설비 ID (예: pv-02)")
    name: str = Field(description="설비 이름 (예: 태양광 2호기)")
    stage: str = Field(description="전력 흐름 단계: 수집 / 생산 / 이동 / 저장 / 공급")
    status: str = Field(description="상태: healthy / degraded / down / unknown")
    metrics: dict[str, int | float] = Field(default_factory=dict, description="계측값 묶음")
    last_checked: str = Field(description="조회 시각 (ISO 8601)")
    alerts: list[str] = Field(default_factory=list, description="활성 알람 목록")


READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _fault_targets() -> set[str]:
    return {x.strip() for x in os.environ.get("DER_FAULT_INJECT", "").split(",") if x.strip()}


@mcp.tool(annotations=READ_ONLY)
def get_equipment_status(equipment_id: str) -> EquipmentStatus:
    """분산에너지 설비 1대의 현재 계측값과 알람을 조회합니다. 설비를 움직이지 않는 읽기 전용 조회입니다.

    입력: equipment_id — 설비 ID 문자열. 등록된 값만 유효: gw-01, pv-01, pv-02, pcc-01, tr-01, ess-01, ess-02,
          load-01, evc-01 (한국어 이름이 아니라 ID를 넣어야 함).
    반환: equipment_id, name, stage(수집/생산/이동/저장/공급), status(healthy 정상/warning 주의/degraded 이상/down/unknown — 설비별 상태 판단 기준으로 계산),
          metrics(계측값 묶음, 키는 단위 접미사 _kw·_percent·_c·_v 등), last_checked(ISO 8601), alerts(알람 문장 목록).
    제약: 등록되지 않은 ID면 오류 대신 status="unknown"과 안내 알람을 돌려줍니다."""
    if equipment_id in _fault_targets():
        raise RuntimeError(f"장애 주입: {equipment_id} 계측 서버 응답 오류")
    equipment = json.loads(MONITORING_FILE.read_text())   # 호출할 때마다 새로 읽음 → 파일을 고치면 바로 반영
    now = datetime.now().isoformat(timespec="seconds")
    if equipment_id not in equipment:
        return EquipmentStatus(
            equipment_id=equipment_id, name="(미등록)", stage="-", status="unknown",
            last_checked=now, alerts=[f"설비 '{equipment_id}'은(는) 등록되지 않은 설비입니다"],
        )
    data = equipment[equipment_id]
    # 상태·알람은 계측값을 설비별 상태 판단 기준(status_criteria)과 비교해 만든다(agent/diagnosis.py).
    # 데이터에 status/alerts가 직접 적혀 있으면(외부 알람 시스템 값으로 보고) 그대로 쓴다. events는 계측값으로는
    # 알 수 없는 외부 알림(출력제어 지시, DR 요청, 스케줄 상충 등)으로 알람 뒤에 붙인다.
    findings = judge({"equipment_id": equipment_id, **data})
    status = data.get("status") or overall_status(findings)
    alerts = data["alerts"] if "alerts" in data else generated_alerts(findings)
    return EquipmentStatus(
        equipment_id=equipment_id,
        name=data["name"],
        stage=data["stage"],
        status=status,
        metrics=data.get("metrics", {}),
        last_checked=now,
        alerts=alerts + data.get("events", []),
    )


@mcp.resource("runbook://der-response/{incident_type}", mime_type="application/json")
def get_incident_runbook(incident_type: str) -> str:
    """이상 유형별 대응 매뉴얼을 반환합니다.

    반환(JSON 문자열): {"incident_type", "content"(마크다운 본문), "last_modified"(ISO 8601), "sha256"(본문 해시),
    "found"(bool)}. 이 문서는 운영 중 수정될 수 있습니다 — 캐시할 때는 last_modified/sha256으로 최신인지 확인하세요."""
    runbook_path = DATA_DIR / "runbooks" / f"{incident_type}.md"
    if not runbook_path.exists():
        content = f"# Runbook Not Found\n\nNo runbook for incident type: {incident_type}"
        return json.dumps({"incident_type": incident_type, "content": content, "last_modified": None,
                           "sha256": None, "found": False}, ensure_ascii=False)
    content = runbook_path.read_text()
    modified = datetime.fromtimestamp(runbook_path.stat().st_mtime).isoformat(timespec="seconds")
    return json.dumps({"incident_type": incident_type, "content": content, "last_modified": modified,
                       "sha256": hashlib.sha256(content.encode()).hexdigest()[:16], "found": True},
                      ensure_ascii=False)


_RUNBOOK_INDEX: RunbookIndex | None = None   # 첫 검색 때 한 번만 만들고 재사용(매뉴얼 임베딩 비용 절약)


@mcp.tool(annotations=READ_ONLY)
def search_runbooks(query: str, top_k: int = 3) -> dict:
    """[After 추가 — RAG] 운영자 질문과 의미가 가장 가까운 대응 매뉴얼을 점수 순으로 돌려줍니다.

    입력: query — 운영자 질문 원문(한국어). top_k — 돌려받을 매뉴얼 수(1~9, 기본 3).
    반환: {"backend": 임베딩 방식(tfidf/ollama/openai), "hits": [{"incident_type", "title", "stage", "score"}]}.
          score는 0~1 코사인 유사도이며 방식마다 크기가 다르므로 기준값은 config/rules.json의 rag.min_score를 쓰세요.
    제약: 매뉴얼 본문은 주지 않습니다 — 본문은 Resource runbook://der-response/{incident_type}으로 읽으세요."""
    global _RUNBOOK_INDEX
    if _RUNBOOK_INDEX is None:
        _RUNBOOK_INDEX = RunbookIndex()
    return {"backend": embedding_backend(), "hits": _RUNBOOK_INDEX.search(query, top_k)}


@mcp.tool(annotations=READ_ONLY)
def get_power_balance() -> dict:
    """[After 추가] 단지 전체 전력 수지를 계산합니다. 공급(발전·방전·수전) 합과 수요(부하·충전·역송) 합이
    같아야 정상이며, 차이가 크면 계측 누락이나 오류를 의심합니다.

    입력: 없음.
    반환: {"supply_kw", "demand_kw", "imbalance_kw"(공급-수요), "supply": [{equipment_id, metric, kw}], "demand": [...]}.
    제약: 단위는 모두 kW. 어떤 계측값을 공급/수요로 볼지는 config/rules.json의 power_balance에서 정합니다."""
    equipment = json.loads(MONITORING_FILE.read_text())
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
