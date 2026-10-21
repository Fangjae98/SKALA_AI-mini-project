"""[After 추가 — RAG + 우선순위 규칙] 어떤 대응 매뉴얼을 줄지 고르는 규칙.

입력
- hits      : RAG 검색 결과(매뉴얼별 '질문 의미 점수')
- suspected : 전문 에이전트들이 계측값·알람으로 실제 확인한 이상 유형(우선순위 순)
- backend   : 임베딩 방식(tfidf / ollama / openai) — 방식마다 점수 크기가 달라 기준값을 따로 쓴다

규칙
1) 최종 점수 = 의미 점수 + (데이터로 확인된 이상이고 의미 점수가 최소 기준 이상이면 가산점)
2) 1등의 최종 점수가 min_score 이상이면 그것이 '주 매뉴얼'(질문 의도에 맞춘 답)
3) 아니면 질문 의도가 불분명하다고 보고 데이터로 확인된 이상 중 우선순위 1등을 주 매뉴얼로
4) 데이터로 확인된 '안전 이슈'(safety)가 주 매뉴얼과 다르면 함께 안내(max_runbooks까지)
"""
from .config import RULES


def choose_runbooks(hits: list[dict], suspected: list[str], backend: str) -> dict:
    rag = RULES["rag"]
    types = RULES["incident_types"]
    min_score = rag["min_score"].get(backend, 0.1)
    ev_min = rag["evidence_min_score"].get(backend, 0.0)
    bonus = rag["evidence_bonus"]

    ranked = []
    for h in hits:
        evidence = h["incident_type"] in suspected and h["score"] >= ev_min
        ranked.append({**h, "evidence": evidence,
                       "final": round(h["score"] + (bonus if evidence else 0.0), 4)})
    ranked.sort(key=lambda r: r["final"], reverse=True)

    if ranked and ranked[0]["final"] >= min_score:
        primary, method = ranked[0]["incident_type"], "rag"
    elif suspected:
        primary, method = suspected[0], "priority"
    elif ranked:
        primary, method = ranked[0]["incident_type"], "rag-low"
    else:
        primary, method = None, "none"

    safety = [t for t in suspected if types.get(t, {}).get("safety") and t != primary]
    extras = safety[: max(0, rag["max_runbooks"] - 1)]
    return {"primary": primary, "extras": extras, "method": method, "ranked": ranked[:3]}
