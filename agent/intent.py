"""질문 3분류 — 잡담 / 설비 조회 / 매뉴얼 조회(이상 대응).

우리 서비스는 분산에너지 운영자에게 대응 매뉴얼을 제공한다. 들어올 수 있는 질문은 세 가지다.
  1) 잡담       : 인사·농담·"뭘 할 수 있어?"·"어떤 매뉴얼이 있어?"
  2) 설비 조회  : "어떤 설비가 있어?", "ESS 1호기 상태 알려줘"
  3) 매뉴얼 조회: "ESS 1호기 온도가 올라가요" → 조치 제안 → 운영자 승인 → 결과 안내

인사·도움말·매뉴얼 목록·설비 조회는 정답이 데이터 파일에 그대로 있으므로 AI 모델을 부르지 않고 코드가 답한다.
(AI가 지어내지 않고, 작은 모델이 오래 생각하느라 멈추는 일도 없다.) 규칙에 안 걸리는 질문만 AI 분류로 넘긴다.
단어 목록은 config/rules.json의 intent에서 읽는다.
"""
import re
from pathlib import Path

from .config import RULES
from .diagnosis import LEVELS, judge, problems, suggested_question
from .equipment import EQUIPMENT, format_metrics

RUNBOOK_DIR = Path(__file__).resolve().parent.parent / "data" / "runbooks"

_CFG = RULES["intent"]
STATUS_LABELS = {"healthy": "정상", "warning": "주의", "degraded": "이상", "down": "중지", "unknown": "확인 불가"}
STAGE_ORDER = ["수집", "생산", "이동", "저장", "공급"]


def _has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def quick_reply_kind(query: str) -> str | None:
    """AI 없이 코드가 바로 답할 잡담 종류: greeting / help / manual_list / None."""
    q = query.strip()
    if _has(_CFG["symptom_pattern"], q):          # 증상이 섞여 있으면 이상 대응 질문으로 본다
        return None
    if _has(_CFG["manual_list_pattern"], q):
        return "manual_list"
    if _has(_CFG["help_pattern"], q):
        return "help"
    if _has(_CFG["greeting_pattern"], q) and len(q) <= _CFG["greeting_max_len"]:
        return "greeting"
    return None


def classify_by_rules(query: str) -> str | None:
    """규칙으로 확실히 분류되면 chat / equipment / incident, 애매하면 None(→ AI 분류).
    증상 단어가 있으면 이상 대응(incident)으로 고정한다 — 작은 AI 모델이 "잔량이 바닥이에요"를
    설비 조회로 착각하는 일을 막는다(안전 쪽으로 기울인 선택: 애매하면 매뉴얼까지 확인)."""
    if quick_reply_kind(query):
        return "chat"
    if _has(_CFG["symptom_pattern"], query):
        return "incident"
    if _has(_CFG["equipment_query_pattern"], query):
        return "equipment"
    return None


# ── 코드가 만드는 답변 ───────────────────────────────────────────────────
def equipment_catalog() -> str:
    """단계별 설비 목록 (상태 없이 이름만) — 잡담 답변 AI에게 넘겨 지어내지 않게 하는 용도."""
    by_stage: dict[str, list[str]] = {}
    for eq_id, info in EQUIPMENT.items():
        by_stage.setdefault(info.get("stage", "-"), []).append(f"{info.get('name', eq_id)}({eq_id})")
    return " / ".join(f"[{s}] {', '.join(by_stage[s])}" for s in STAGE_ORDER if s in by_stage)


def runbook_titles() -> list[str]:
    titles = []
    for path in sorted(RUNBOOK_DIR.glob("*.md")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        name = re.sub(r"^#\s*Runbook:\s*", "", first).split(" 대응")[0].strip()
        titles.append(f"{name}")
    return titles


def quick_reply(kind: str) -> str:
    if kind == "greeting":
        return ("안녕하세요! 분산에너지 운영 도우미입니다.\n"
                "• 설비 조회: \"어떤 설비가 있어?\", \"ESS 1호기 상태 알려줘\"\n"
                "• 이상 대응: \"ESS 1호기 온도가 계속 올라가요\"처럼 문제 상황을 말씀해 주세요.")
    if kind == "manual_list":
        lines = "\n".join(f"{i}. {t}" for i, t in enumerate(runbook_titles(), 1))
        return (f"대응 매뉴얼 {len(runbook_titles())}종이 있습니다.\n{lines}\n\n"
                "문제 상황을 말씀해 주시면 설비 데이터를 확인해 맞는 매뉴얼의 조치를 안내합니다.")
    return ("제가 할 수 있는 일은 세 가지입니다.\n"
            "1) 설비 조회 — 연결된 설비 목록과 현재 상태 (예: \"어떤 설비가 있어?\", \"변압기 상태 알려줘\")\n"
            "2) 이상 대응 — 문제 상황을 말씀하시면 설비 데이터를 확인하고 대응 매뉴얼의 조치를 안내합니다. "
            "설비를 움직이는 조치는 운영자 승인 후에 반영합니다. (예: \"ESS 1호기 온도가 계속 올라가요\")\n"
            "3) 매뉴얼 목록 — \"어떤 매뉴얼이 있어?\"\n"
            f"\n연결된 설비: {equipment_catalog()}")


def runbook_title_of(incident_type: str) -> str:
    path = RUNBOOK_DIR / f"{incident_type}.md"
    if not path.exists():
        return incident_type
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return re.sub(r"^#\s*Runbook:\s*", "", first).split(" 대응")[0].strip()


def format_overview(statuses: list[dict], failures: list[str]) -> str:
    """설비 전체 현황 — 상태별 대수 요약 → 문제 설비 → 단계별 목록."""
    counts: dict[str, int] = {}
    for s in statuses:
        label = STATUS_LABELS.get(s["status"], s["status"])
        counts[label] = counts.get(label, 0) + 1
    if failures:
        counts["조회 실패"] = len(failures)
    summary = " · ".join(f"{k} {v}대" for k, v in counts.items())
    lines = [f"현재 연결된 설비 {len(statuses) + len(failures)}대 — {summary}"]
    bad = [s for s in statuses if s["status"] != "healthy"]
    if bad:
        lines.append("문제 설비: " + ", ".join(f"{s['name']}({s['equipment_id']})" for s in bad))
    lines.append("")
    for stage in STAGE_ORDER:
        for s in [x for x in statuses if x.get("stage") == stage]:
            label = STATUS_LABELS.get(s["status"], s["status"])
            alerts = s.get("alerts", [])
            tail = f" — {alerts[0]}{f' 외 {len(alerts) - 1}건' if len(alerts) > 1 else ''}" if alerts else ""
            lines.append(f"[{stage}] {s['name']}({s['equipment_id']}): {label}{tail}")
    for eq in failures:
        lines.append(f"[-] {eq}: 조회 실패 — 데이터를 받지 못했습니다")
    lines.append("\n설비별 분석은 \"태양광 2호기 상태 알려줘\"처럼 물어보세요.")
    return "\n".join(lines)


def format_detail(s: dict) -> str:
    """설비 1대 상태 분석 — ① 계측값 ② 기준 대비 판정 ③ 문제점 분석 ④ 다음 단계."""
    findings = judge(s)
    label = STATUS_LABELS.get(s["status"], s["status"])
    probs = problems(findings)
    head = (f"{s['name']}({s['equipment_id']}) · {s.get('stage', '-')} 단계 · 상태: {label}"
            + (f" (문제 {len(probs)}건)" if probs else ""))
    parts = [head, "", "① 계측값", f"- {format_metrics(s.get('metrics', {})) or '없음'} (확인 시각 {s.get('last_checked', '-')})"]
    parts += ["", "② 상태 판단 (설비별 기준 대비)"]
    judged = [f for f in findings if f["level"] != "info"]
    if judged:
        for f in judged:
            rule = f" [기준: {f['rule']}]" if f.get("rule") else ""
            parts.append(f"- {f['label']} {f['value']}{_unit(f)} → {LEVELS[f['level']]}{rule}")
    else:
        parts.append("- 이 설비 종류는 판단 기준 항목이 없습니다 (계측값만 표시)")
    parts += ["", "③ 문제점 분석"]
    if probs:
        for f in probs:
            suspect = f" (의심 이상: {runbook_title_of(f['incident'])})" if f.get("incident") else ""
            parts.append(f"- {f['meaning']}{suspect}")
    else:
        parts.append("- 기준을 벗어난 항목이 없습니다.")
    for f in findings:
        if f["level"] in ("hold", "info") and f.get("meaning"):
            parts.append(f"- (참고) {f['meaning']}")
    generated = {f.get("alert") for f in findings}
    events = [a for a in s.get("alerts", []) if a not in generated]   # 기준 판정이 아닌 설비 알람·외부 알림
    if events:
        parts += ["", "설비 알람·외부 알림"] + [f"- {e}" for e in events]
    q = suggested_question(s, findings)
    if q:
        parts += ["", "④ 다음 단계", f"대응 방법이 필요하면 이렇게 물어보세요: \"{q}\""]
    return "\n".join(parts)


def _unit(f: dict) -> str:
    from .config import RULES
    for c in RULES["status_criteria"].values():
        for item in c:
            if item["metric"] == f["metric"]:
                return item.get("unit", "")
    return ""
