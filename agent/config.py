"""[After 추가 — 모듈화] config/rules.json(판단 기준 설정 파일)을 읽어 모든 모듈이 같은 값을 쓰게 한다.

Before에서는 기본 설비, 키워드, 우선순위, 검수 기준이 graph.py·fake_llm.py·equipment.py 등에
흩어져 있었다. 이제 기준을 바꿀 때는 config/rules.json만 고치면 된다.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("DER_RULES_PATH", ROOT / "config" / "rules.json"))


def _strip_notes(value):
    """'_설명'처럼 밑줄로 시작하는 사람용 메모 항목을 제거한다."""
    if isinstance(value, dict):
        return {k: _strip_notes(v) for k, v in value.items() if not k.startswith("_")}
    return value


@lru_cache(maxsize=1)
def load_rules() -> dict:
    return _strip_notes(json.loads(CONFIG_PATH.read_text()))


RULES = load_rules()


def incident_types_by_priority() -> list[str]:
    """이상 유형을 우선순위(숫자 작을수록 먼저) 순서로 돌려준다."""
    types = RULES["incident_types"]
    return sorted(types, key=lambda t: types[t]["priority"])


def types_from_alerts(alerts: list[str]) -> list[str]:
    """알람 문구들에 등장하는 키워드로 이상 유형을 찾아 우선순위 순서로 돌려준다."""
    text = " ".join(alerts).lower()
    return [t for t in incident_types_by_priority()
            if any(k.lower() in text for k in RULES["incident_types"][t]["alert_keywords"])]


def embedding_backend() -> str:
    """RAG 임베딩 방식: rules.json의 rag.embedding_backend, auto면 LLM_BACKEND를 따른다."""
    configured = os.environ.get("EMBEDDING_BACKEND") or RULES["rag"]["embedding_backend"]
    if configured != "auto":
        return configured
    llm = os.environ.get("LLM_BACKEND", "fake")
    return {"ollama": "ollama", "openai": "openai"}.get(llm, "tfidf")
