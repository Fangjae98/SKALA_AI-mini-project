"""설비 상태 진단 — 설비별 '상태 판단 기준'(config/rules.json의 status_criteria)으로 계측값을 판정한다.

진단 순서
  ① 계측값 읽기          : MCP get_equipment_status (data/monitoring.json을 매번 새로 읽음)
  ② 기준 대비 판정        : 항목마다 정상 / 주의 / 경보 (보조 조건이 안 맞으면 '판정 보류')
  ③ 설비 상태 결정        : 경보가 하나라도 있으면 '이상', 주의만 있으면 '주의', 아니면 '정상'
  ④ 문제점 분석           : 주의·경보 항목이 뜻하는 것(meaning)과 의심되는 이상 유형(incident)

숫자 판단은 전부 코드가 한다(AI에 맡기지 않음). 기준값을 바꾸려면 rules.json만 고치면 된다.
설비 종류는 설비 ID 앞부분(gw, pv, pcc, tr, ess, load, evc)으로 정한다.
"""
from .config import RULES

LEVELS = {"normal": "정상", "warn": "주의", "alarm": "경보", "hold": "판정 보류", "info": "참고"}


def equipment_kind(equipment_id: str) -> str:
    return equipment_id.split("-")[0]


def _derived(m: dict) -> dict:
    """여러 계측값을 조합해야 판단할 수 있는 값(비율·여유)을 미리 계산해 둔다."""
    d = dict(m)
    if m.get("total_devices"):
        d["connected_ratio_percent"] = round(m.get("connected_devices", 0) / m["total_devices"] * 100, 1)
        d["missing_devices"] = m["total_devices"] - m.get("connected_devices", 0)
    if m.get("contract_kw") and "today_peak_kw" in m:
        d["peak_ratio_percent"] = round(m["today_peak_kw"] / m["contract_kw"] * 100, 1)
    if m.get("rated_voltage_v") and "bus_voltage_v" in m:
        d["voltage_rise_percent"] = round(m["bus_voltage_v"] / m["rated_voltage_v"] * 100 - 100, 1)
    if "soc_percent" in m:
        if "soc_upper_limit" in m:
            d["soc_upper_margin"] = round(m["soc_upper_limit"] - m["soc_percent"], 1)
        if "soc_lower_limit" in m:
            d["soc_lower_margin"] = round(m["soc_percent"] - m["soc_lower_limit"], 1)
    return d


def _hit(value: float, limit: float, direction: str, strict: bool) -> bool:
    if direction == "high":
        return value > limit if strict else value >= limit
    return value < limit if strict else value <= limit


def _fmt(template: str, values: dict) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return template


def judge(status: dict) -> list[dict]:
    """설비 1대의 기준 항목별 판정 목록. 계측값이 없는 항목은 건너뛴다."""
    eq = status["equipment_id"]
    criteria = RULES["status_criteria"].get(equipment_kind(eq), [])
    values = {"eq": eq, "name": status.get("name", eq), **_derived(status.get("metrics", {}))}
    out = []
    for c in criteria:
        key = c["metric"]
        if key not in values:
            continue
        v = values[key]
        fields = {**values, "v": v, "unit": c.get("unit", ""), "warn": c.get("warn"), "alarm": c.get("alarm")}
        req = c.get("requires")
        if req and values.get(req["metric"], 0) < req["min"]:
            out.append({"metric": key, "label": c["label"], "value": v, "level": "hold", "incident": None,
                        "unit": c.get("unit", ""), "criterion_incident": c.get("incident"),
                        "check": None, "alert": None, "meaning": _fmt(req["reason"], fields),
                        "rule": _fmt(c.get("rule", ""), fields)})
            continue
        if c.get("info_when_positive"):
            if v:
                out.append({"metric": key, "label": c["label"], "value": v, "level": "info", "incident": None,
                            "check": _fmt(c["check"], fields), "alert": None, "meaning": _fmt(c.get("meaning", ""), fields),
                            "rule": ""})
            continue
        level = "normal"
        strict = c.get("strict", False)
        if c.get("alarm") is not None and _hit(v, c["alarm"], c["direction"], strict):
            level = "alarm"
        elif c.get("warn") is not None and _hit(v, c["warn"], c["direction"], strict):
            level = "warn"
        fields["limit"] = c.get(level) if level in ("warn", "alarm") else c.get("warn")
        fields["level"] = LEVELS[level]
        check_t = c.get("warn_check") if level == "warn" and c.get("warn_check") else c.get("check", "")
        out.append({"metric": key, "label": c["label"], "value": v, "level": level, "unit": c.get("unit", ""),
                    "criterion_incident": c.get("incident"),
                    "incident": c.get("incident") if level != "normal" else None,
                    "check": _fmt(check_t, fields) if level != "normal" else None,
                    "alert": _fmt(c.get("alert", ""), fields) if level != "normal" else None,
                    "meaning": _fmt(c.get("meaning", ""), fields) if level != "normal" else "",
                    "rule": _fmt(c.get("rule", ""), fields)})
    return out


def overall_status(findings: list[dict]) -> str:
    levels = {f["level"] for f in findings}
    if "alarm" in levels:
        return "degraded"
    if "warn" in levels:
        return "warning"
    return "healthy"


def generated_alerts(findings: list[dict]) -> list[str]:
    """주의·경보 항목을 알람 문장으로 — 데이터 파일에 알람을 손으로 적지 않아도 기준으로 만들어진다."""
    order = {"alarm": 0, "warn": 1}
    return [f["alert"] for f in sorted(findings, key=lambda f: order.get(f["level"], 9))
            if f["level"] in order and f["alert"]]


def problems(findings: list[dict]) -> list[dict]:
    order = {"alarm": 0, "warn": 1}
    return sorted([f for f in findings if f["level"] in order], key=lambda f: order[f["level"]])


def evidence_for(collected: list[dict], incident: str) -> list[str]:
    """그 이상 유형을 판단하는 기준 항목의 현재 판정 — '데이터로 확인되는가'를 보여 주는 근거 문장."""
    lines = []
    for c in collected:
        if c.get("source") != "monitoring":
            continue
        d = c["data"]
        for f in judge(d):
            if f.get("criterion_incident") == incident and f["level"] not in ("info",):
                lines.append(f"{d['equipment_id']} {f['label']} {f['value']}{f.get('unit', '')} → {LEVELS[f['level']]}"
                             + (f" (기준: {f['rule']})" if f.get("rule") else ""))
    return lines


def suggested_question(status: dict, findings: list[dict]) -> str | None:
    """'대응 요청' 버튼용 질문 — 알람 문장을 그대로 복사하지 않고, 진단 결과를 운영자 말투로 정리한다."""
    probs = problems(findings)
    if not probs:
        return None
    facts = ", ".join(f["check"].replace(f"{status['equipment_id']} ", "", 1) for f in probs[:2] if f["check"])
    return f"{status.get('name', status['equipment_id'])}({status['equipment_id']})에서 {facts}. 원인과 대응 방법을 알려주세요."
