"""분산에너지 설비 정보 도우미 — data/monitoring.json 하나를 기준으로 설비 목록,
별칭(한국어 이름), 측정값 표시 형식을 제공한다.

Base에서는 서버 이름(auth-service 등)을 영어 ID로만 찾았다. 분산에너지 운영자는
"태양광 2호기", "ESS", "변압기"처럼 한국어로 말하므로, 설비마다 별칭(aliases)을 두고
질문 속에서 ID나 별칭이 등장하는 위치로 설비를 찾는다.
"""
import json
import os
from pathlib import Path

from .config import RULES

ROOT_DIR = Path(__file__).resolve().parent.parent
# 설비 데이터 파일. 기본은 화면 시연용 data/monitoring.json.
# 테스트·회귀 평가는 정답이 고정된 시나리오(data/scenarios/full_incident.json)를 DER_MONITORING_FILE로 지정한다.
DATA_PATH = ROOT_DIR / os.environ.get("DER_MONITORING_FILE", "data/monitoring.json")

# [After 변경 — 모듈화] 아래 값들은 config/rules.json에서 읽는다(Before에서는 이 파일에 직접 적혀 있었음).
DEFAULT_EQUIPMENT = RULES["equipment"]["default"]
REFERENCE_EQUIPMENT = RULES["equipment"]["reference_for_fake"]
GROUP_ALIASES = RULES["equipment"]["group_aliases"]
METRIC_LABELS = {k: tuple(v) for k, v in RULES["metric_labels"].items()}


def load_equipment() -> dict:
    try:
        return json.loads(DATA_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


EQUIPMENT = load_equipment()
KNOWN_SYSTEMS = set(EQUIPMENT.keys()) or {DEFAULT_EQUIPMENT}


def find_mentions(query: str) -> list[str]:
    """질문에 등장하는 설비 ID 목록을 '처음 등장한 위치' 순서로 돌려준다.

    1) 설비 ID(pv-02)와 개별 별칭(태양광 2호기)을 먼저 찾고
    2) 묶음 별칭(태양광, ESS)은 그 묶음의 개별 설비가 하나도 안 잡혔을 때만 쓴다.
    대소문자는 구분하지 않는다."""
    q = query.lower()
    found: dict[str, int] = {}
    for eq_id, info in EQUIPMENT.items():
        names = [eq_id] + info.get("aliases", [])
        positions = [q.index(n.lower()) for n in names if n.lower() in q]
        if positions:
            found[eq_id] = min(positions)
    for alias, members in GROUP_ALIASES.items():
        alias = alias.lower()
        if alias in q and not any(m in found for m in members):
            for m in members:
                if m in EQUIPMENT:
                    found[m] = q.index(alias)
    return [eq_id for eq_id, _ in sorted(found.items(), key=lambda kv: kv[1])]


def format_metrics(metrics: dict) -> str:
    """{"output_kw": 96, "pr_percent": 46} → "출력 96kW, 발전 효율(PR) 46%" """
    parts = []
    for key, value in metrics.items():
        label, unit = METRIC_LABELS.get(key, (key, ""))
        parts.append(f"{label} {value}{unit}")
    return ", ".join(parts)


def describe(status: dict) -> str:
    """MCP Tool 응답(설비 상태 1건)을 보고서용 한 줄로 바꾼다."""
    return (f"{status['equipment_id']}({status.get('name', '')}, {status.get('stage', '')} 단계): "
            f"상태={status['status']}, {format_metrics(status.get('metrics', {}))}, "
            f"알람 {len(status.get('alerts', []))}건 {status.get('alerts', [])}")


def stage_of(equipment_id: str) -> str:
    return EQUIPMENT.get(equipment_id, {}).get("stage", "")
