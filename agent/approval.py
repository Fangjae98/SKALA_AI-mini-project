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


def extract_control_actions(draft: str) -> list[str]:
    cfg = RULES["approval"]
    actions = []
    for s in split_sentences(draft):
        if cfg["marker"] in s or any(p in s for p in cfg["action_phrases"]):
            if s not in actions:
                actions.append(s)
    return actions


def approval_footer(actions: list[str], approved: bool, comment: str = "") -> str:
    if approved:
        head = f"[운영자 승인 완료] 위 제어 조치 {len(actions)}건을 승인했습니다."
    else:
        head = (f"[운영자 거부] 위 제어 조치 {len(actions)}건은 시행하지 않습니다. "
                "점검·기록 등 설비를 움직이지 않는 조치만 진행하세요.")
    return head + (f" (메모: {comment})" if comment else "")
