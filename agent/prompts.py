"""[After 추가 — 모듈화] LLM 지시문(프롬프트)과 '수집된 사실' 정리 방식을 한 곳에 모은 파일.

Before에서는 ollama_llm.py와 openai_llm.py에 같은 지시문이 두 번씩 복사돼 있어서, 한쪽만
고치면 두 백엔드의 동작이 달라지는 문제가 있었다. 이제 두 백엔드가 이 파일을 함께 쓴다.
"""
from .config import RULES
from .equipment import describe

APPROVAL_MARKER = RULES["approval"]["marker"]

ROUTE_SYSTEM = (
    "당신은 분산에너지(태양광·ESS·연계점·EV충전기 등) 운영 지원 라우터다. 사용자 요청을 분류하라. "
    "설비 이상·출력 저하·피크·전압·충전율·과열·통신 두절·출력제어·수요반응(DR) 등 운영 이슈면 "
    "route=incident, 인사·잡담·일반 질문이면 route=chat. "
    "needs_analysis는 여러 설비 사이의 연쇄 영향 비교가 꼭 필요할 때만 true."
)


def plan_system(known_ids: list[str]) -> str:
    return (
        "당신은 분산에너지 운영 Supervisor의 planning 노드다. 표준 진단 절차에 따라 실행 계획(tasks)을 "
        "JSON으로 세워라. 사용자 요청에 등장한 설비마다 agent=research task를 하나씩 세우고, target에는 "
        f"반드시 설비 ID를 넣어라 (알려진 설비 ID: {', '.join(known_ids)} — 예: 태양광 2호기=pv-02, "
        "ESS 1호기=ess-01, 변압기=tr-01, 연계점=pcc-01. 특정 설비가 언급되지 않았으면 "
        f"{RULES['equipment']['default']}을 기본으로 조사하라). "
        "여러 설비 사이의 연쇄 영향 분석이 꼭 필요하다고 판단되면 agent=analysis task를 추가해도 된다."
    )


ANALYZE_SYSTEM = (
    "당신은 분산에너지 운영 분석가다. 여러 설비의 계측 결과를 비교해 생산·이동·저장·공급 단계 사이의 "
    "연쇄 영향이나 공통 원인이 있는지 한두 문장으로 짧게 답하라. 수치는 주어진 값만 쓰라. "
    "패턴이 없으면 '특이 패턴 없음'이라고만 답하라."
)

WRITE_SYSTEM = (
    "당신은 분산에너지 운영 보고 작성자다. 수집된 사실만 사용해 한국어로 간결한 대응 보고를 작성하라. "
    "반드시 '1) 현재 상태 2) 원인 추정 3) 권장 조치' 형식의 번호로 답하고, 수치는 수집된 값을 단위까지 "
    "그대로 인용하라(계산하거나 지어낸 숫자 금지). 안전 이슈 매뉴얼이 함께 주어지면 권장 조치에서 먼저 다뤄라. "
    f"설비를 움직이는 조치(충방전·출력 제한·부하 차단 등)에는 문장 끝에 '{APPROVAL_MARKER}'를 붙여라."
)

CHAT_SYSTEM = (
    "당신은 분산에너지 단지 운영을 돕는 간결한 한국어 비서다. 한두 문장으로 답하라. "
    "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라."
)


def analyze_facts(collected: list[dict]) -> str:
    mon = [c["data"] for c in collected if c["source"] == "monitoring"]
    return "\n".join(f"- {describe(d)}" for d in mon) or "(계측 데이터 없음)"


def write_facts(collected: list[dict]) -> str:
    """write 노드에 넘길 '수집된 사실' — 설비 계측, 전문 에이전트 점검, 매뉴얼, 연쇄 분석 순서."""
    facts = []
    for c in collected:
        if c["source"] == "monitoring":
            facts.append(f"- 설비 계측 {describe(c['data'])}")
    for c in collected:
        if c["source"] == "specialist":
            d = c["data"]
            checks = "; ".join(d["checks"]) or "이상 없음"
            facts.append(f"- [{d['agent']}] 점검: {checks}")
    for c in collected:
        if c["source"] == "runbook":
            label = "안전 이슈 매뉴얼" if c.get("role") == "safety" else "대응 매뉴얼"
            facts.append(f"- {label} 발췌:\n{c['data'].strip()}")
        elif c["source"] == "analysis":
            facts.append(f"- 연쇄 분석: {c['data'].strip()}")
    return "\n".join(facts)
