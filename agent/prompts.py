"""[After 추가 — 모듈화] LLM 지시문(프롬프트)과 '수집된 사실' 정리 방식을 한 곳에 모은 파일.

Before에서는 ollama_llm.py와 openai_llm.py에 같은 지시문이 두 번씩 복사돼 있어서, 한쪽만
고치면 두 백엔드의 동작이 달라지는 문제가 있었다. 이제 두 백엔드가 이 파일을 함께 쓴다.
"""
from .config import RULES
from .diagnosis import evidence_for
from .equipment import describe
from .intent import equipment_catalog

APPROVAL_MARKER = RULES["approval"]["marker"]

ROUTE_SYSTEM = (
    "당신은 분산에너지(태양광·ESS·연계점·EV충전기 등) 운영 지원 라우터다. 사용자 요청을 분류하라. "
    "설비 이상·출력 저하·피크·전압·충전율·과열·통신 두절·출력제어·수요반응(DR) 등 문제 상황을 알리거나 대응 방법을 묻으면 "
    "route=incident, 설비 목록이나 특정 설비의 현재 상태·값만 묻으면 route=equipment, "
    "인사·잡담·일반 질문이면 route=chat. "
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
    f"설비를 움직이는 조치(충전·방전의 중지·시작·전환, 출력 제한·조정, 부하 차단, 재기동, 분리 등)는 조치 하나를 한 문장으로 쓰고 "
    f"문장 끝에 정확히 '{APPROVAL_MARKER}'를 괄호까지 그대로 붙여라. 점검·확인·기록 문장에는 붙이지 마라."
)

# 운영자 판단이 있을 때만 작성 지시에 덧붙인다(없을 때 넣으면 모델이 '운영자 판단 반영' 항목을 지어냄 — Ollama 실측)
OPERATOR_REWRITE_RULE = (
    "## 운영자 판단 반영 (반드시 지킬 것)\n"
    "위 [운영자 판단]은 운영자가 현장을 보고 내린 결정이다. 이전 보고서를 고쳐 쓰지 말고, 이 판단을 근거로 문제를 다시 분석해 "
    "원인 추정과 권장 조치를 새로 구성하라. 판단이 매뉴얼과 다르면 판단을 따르되, 안전 이슈 매뉴얼의 필수 안전 조치를 빼야 하면 "
    "빼지 말고 이유를 적어라. 운영자가 지시한 설비 조치도 조치마다 한 문장으로 쓰고 끝에 "
    f"'{APPROVAL_MARKER}'를 붙여라. 마지막 번호 항목은 '운영자 판단 반영: 무엇을 어떻게 바꿨는지'로 쓴다."
)


def operator_rule(collected: list[dict]) -> str:
    return "\n\n" + OPERATOR_REWRITE_RULE if any(c["source"] == "operator" for c in collected) else ""

JUDGE_SYSTEM = (
    "당신은 분산에너지 운영 보고서를 검수하는 심판이다. 보고서를 쓴 작성자와는 별개의 검수자로서, "
    "사용자 요청·수집된 사실·보고서 초안을 보고 다음 4가지 기준으로 판정하라. "
    "Relevance(관련성): 사용자가 물은 설비·이상에 답하는가. "
    "Completeness(완전성): 수집된 대응 매뉴얼(특히 '안전 이슈 매뉴얼')의 핵심 조치가 빠짐없이 들어 있는가. "
    "Clarity(명확성): 1) 2) 3) 번호 형식으로 운영자가 바로 이해할 수 있는가. "
    "Domain(도메인 적합성): '이번 보고서의 안전 이슈 매뉴얼' 목록에 있는 조치를 다른 조치보다 먼저 다루고, "
    f"설비를 움직이는 조치(충방전 중지·출력 제한·부하 차단·재기동·분리 등)에 '{APPROVAL_MARKER}' 표시가 있는가. "
    "안전 이슈 매뉴얼 목록이 '없음'이면 안전 우선 기준은 적용하지 말고, 일반 대응 매뉴얼의 절차를 안전 이슈로 착각하지 마라. "
    "점검·확인·기록처럼 설비를 움직이지 않는 조치에는 승인 표시가 필요 없다. "
    "사용자 요청에 [운영자 판단]이 있으면 운영자가 현장을 보고 내린 결정이므로, 그 판단에 따라 매뉴얼 조치를 바꾸거나 뺀 것은 "
    "감점하지 마라(단, 안전 이슈 매뉴얼의 필수 안전 조치를 이유 없이 뺐다면 불합격). "
    "사용자가 '짧게' 등을 요구해도 안전 이슈 매뉴얼의 조치를 빼면 불합격이다. 사소한 표현 차이로는 불합격 주지 마라. "
    "reasoning은 반드시 한국어로 쓴다. "
    '반드시 JSON으로만 답하라: {"is_acceptable": true 또는 false, "reasoning": "불합격이면 무엇을 고쳐야 하는지 한두 문장"}'
)

CHAT_SYSTEM = (
    "당신은 분산에너지 단지 운영을 돕는 간결한 한국어 비서다. 한두 문장으로 답하라. "
    "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라. "
    f"이 단지에 연결된 설비는 다음뿐이다: {equipment_catalog()}. 이 목록에 없는 설비를 지어내지 마라. "
    "날씨·뉴스처럼 단지 운영과 무관한 질문에는 그 정보는 확인할 수 없다고 말하고, "
    "대신 도와줄 수 있는 일(설비 상태 조회, 이상 상황 대응 안내)을 한 문장으로 소개하라. "
    "설비 수치는 절대 지어내지 마라."
)


def analyze_facts(collected: list[dict]) -> str:
    mon = [c["data"] for c in collected if c["source"] == "monitoring"]
    return "\n".join(f"- {describe(d)}" for d in mon) or "(계측 데이터 없음)"


def failure_lines(collected: list[dict]) -> list[str]:
    """[개선 I-1] 조회에 실패한 설비·매뉴얼 — 보고서가 '데이터를 못 받은 부분'을 숨기지 않게 한다."""
    return [f"{c['data']['target']} 조회 실패({c['data']['code']}: {c['data']['message']})"
            for c in collected if c["source"] == "failure"]


def write_facts(collected: list[dict]) -> str:
    """write 노드에 넘길 '수집된 사실' — 설비 계측, 전문 에이전트 점검, 매뉴얼, 연쇄 분석, 조회 실패 순서."""
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
            if c.get("unconfirmed"):
                ev = "; ".join(evidence_for(collected, c.get("incident_type", ""))) or "관련 계측 항목 없음"
                facts.append(f"- [데이터 확인] 질문한 이상이 계측값으로 확인되지 않았다({ev}). 설비를 움직이는 조치"
                             "(운영자 승인 필요)는 쓰지 말고, 데이터와 현장 상황이 다른 이유(센서·통신 오류)와 현장 확인 절차를 먼저 안내하라.")
        elif c["source"] == "analysis":
            facts.append(f"- 연쇄 분석: {c['data'].strip()}")
        elif c["source"] == "operator":
            facts.append(f"- [운영자 판단] {c['data'].strip()} — 운영자의 현장 판단이므로 매뉴얼보다 우선 반영하라")
    for line in failure_lines(collected):
        facts.append(f"- [조회 실패] {line} — 이 부분은 데이터가 없으므로 추정하지 말고 '확인 불가'로 적어라")
    return "\n".join(facts)


def _runbook_title(text: str) -> str:
    first = next((l for l in text.strip().splitlines() if l.strip()), "")
    return first.lstrip("# ").replace("Runbook: ", "").strip()


def judge_user(query: str, collected: list[dict], draft: str) -> str:
    """[개선 I-4 보정] 어떤 매뉴얼이 '안전 이슈'인지 심판에게 명시한다 — OpenAI 실측에서 gpt-4o-mini 심판이
    안전 이슈가 없는 질문(피크)에서도 일반 절차를 안전 이슈로 보고 3회 연속 반려한 문제를 막기 위함."""
    safety = [_runbook_title(c["data"]) for c in collected if c["source"] == "runbook" and c.get("role") == "safety"]
    primary = [_runbook_title(c["data"]) for c in collected if c["source"] == "runbook" and c.get("role") != "safety"]
    return (f"## 사용자 요청\n{query}\n\n"
            f"## 이번 보고서의 주 대응 매뉴얼\n{', '.join(primary) or '없음'}\n"
            f"## 이번 보고서의 안전 이슈 매뉴얼\n{', '.join(safety) or '없음'}\n\n"
            f"## 수집된 사실\n{write_facts(collected)}\n\n"
            f"## 검수할 보고서 초안\n{draft}")
