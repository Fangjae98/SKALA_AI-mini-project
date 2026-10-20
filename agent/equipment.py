"""분산에너지 설비 정보 도우미 — data/monitoring.json 하나를 기준으로 설비 목록,
별칭(한국어 이름), 측정값 표시 형식을 제공한다.

Base에서는 서버 이름(auth-service 등)을 영어 ID로만 찾았다. 분산에너지 운영자는
"태양광 2호기", "ESS", "변압기"처럼 한국어로 말하므로, 설비마다 별칭(aliases)을 두고
질문 속에서 ID나 별칭이 등장하는 위치로 설비를 찾는다.
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "monitoring.json"

# 설비를 하나도 못 찾았을 때 기본으로 조사할 설비 — 단지 전체 전력이 지나가는 연계점
DEFAULT_EQUIPMENT = "pcc-01"
# fake 백엔드가 비교용으로 항상 함께 조회하는 설비(Base의 web-01 역할) — 정상 상태인 단지 부하
REFERENCE_EQUIPMENT = "load-01"

# 여러 설비를 한 번에 가리키는 묶음 별칭 — 같은 묶음의 개별 설비가 이미 언급됐으면 쓰지 않는다
GROUP_ALIASES = {
    "태양광": ["pv-01", "pv-02"],
    "pv": ["pv-01", "pv-02"],
    "발전": ["pv-01", "pv-02"],
    "출력제어": ["pv-01", "pv-02"],
    "ess": ["ess-01", "ess-02"],
    "배터리": ["ess-01", "ess-02"],
    "저장장치": ["ess-01", "ess-02"],
}

# 측정값 키 → (한국어 이름, 단위)
METRIC_LABELS = {
    "output_kw": ("출력", "kW"),
    "capacity_kw": ("설비용량", "kW"),
    "irradiance_wm2": ("일사량", "W/m²"),
    "pr_percent": ("발전 효율(PR)", "%"),
    "import_kw": ("수전", "kW"),
    "export_kw": ("역송", "kW"),
    "contract_kw": ("계약전력", "kW"),
    "today_peak_kw": ("오늘 15분 최대수요", "kW"),
    "load_ratio_percent": ("부하율", "%"),
    "bus_voltage_v": ("모선 전압", "V"),
    "rated_voltage_v": ("정격 전압", "V"),
    "capacity_kwh": ("용량", "kWh"),
    "soc_percent": ("충전율(SOC)", "%"),
    "soc_upper_limit": ("충전 상한", "%"),
    "soc_lower_limit": ("방전 하한", "%"),
    "charge_kw": ("충전", "kW"),
    "discharge_kw": ("방전", "kW"),
    "cell_temp_max_c": ("셀 최고온도", "°C"),
    "demand_kw": ("부하", "kW"),
    "charging_kw": ("충전 전력", "kW"),
    "ports_in_use": ("사용 중 충전구", "개"),
    "ports_total": ("전체 충전구", "개"),
    "data_age_min": ("데이터 경과", "분"),
    "connected_devices": ("연결 설비", "대"),
    "total_devices": ("전체 설비", "대"),
    "last_data_min_ago": ("마지막 수신 경과", "분"),
    "missing_rate_percent": ("결측률", "%"),
}


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
