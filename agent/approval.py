"""[After 추가 — HITL(사람 승인)] 보고서에서 '설비를 움직이는 조치' 문장을 찾아낸다.

기준은 config/rules.json의 approval 항목:
- marker(기본 '(운영자 승인 필요)')가 붙은 문장, 또는
- action_phrases(충전 중지, 부하 차단, 출력 제한 …) 중 하나라도 들어 있는 문장
"""
import re

from .config import RULES


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"\n+|(?<=[.。])\s+|\s(?=\d\)\s)", text)
    return [p.strip() for p in parts if p and p.strip()]


# 실모델은 표시를 정확히 따르지 않는다("~하되, 운영자 승인 필요." — Ollama 실측). 괄호·띄어쓰기가 달라도 잡는다.
MARKER_LOOSE = re.compile(r"운영자\s*(?:의\s*)?승인\s*(?:이\s*)?필요")


def extract_control_actions(draft: str) -> list[str]:
    """설비를 움직이는 조치 문장. 놓치는 것(승인 없이 실행)이 더 위험하므로 넓게 잡는다 — 표시가 있거나 조치 표현이 있으면."""
    cfg = RULES["approval"]
    actions = []
    for s in split_sentences(draft):
        if re.match(r"^[-\s\d)]*운영자 판단 반영", s):   # 판단을 어떻게 반영했는지 적은 요약 문장 — 조치가 아님
            continue
        if cfg["marker"] in s or MARKER_LOOSE.search(s) or any(p in s for p in cfg["action_phrases"]):
            if s not in actions:
                actions.append(s)
    return actions


def approval_footer(actions: list[str], approved: bool, comment: str = "",
                    chosen: list[dict] | None = None, excluded: list[str] | None = None) -> str:
    if approved and chosen is not None and (excluded or any(c["text"] != c["original"] for c in chosen)):
        edited = sum(1 for c in chosen if c["text"] != c["original"])
        head = (f"[운영자 승인 완료] 제어 조치 {len(chosen)}건을 승인했습니다"
                + (f"(문구 수정 {edited}건)" if edited else "")
                + (f", {len(excluded)}건은 제외했습니다(시행하지 않음)" if excluded else "") + ".")
    elif approved:
        head = f"[운영자 승인 완료] 위 제어 조치 {len(actions)}건을 승인했습니다."
    else:
        head = (f"[운영자 거부] 위 제어 조치 {len(actions)}건은 시행하지 않습니다. "
                "점검·기록 등 설비를 움직이지 않는 조치만 진행하세요.")
    return head + (f" (메모: {comment})" if comment else "")
