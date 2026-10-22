"""Wave 4/10a — 결정론적 Fake LLM 2종 (fast/quality) + 실측 비용 계산.

실제 LLM 호출 없이도 파이프라인 전체가 결정론적으로 재현되도록, 프롬프트 내용에 따라
정해진 응답을 돌려준다. 각 호출은 LangSmith에 run_type="llm"으로 기록되고, 문자열
길이 기반 토큰 추정치로 비용을 계산해 상태에 누적한다 (Wave 10a의 모델 분리 근거).

[개선 I-2] route/plan/analyze를 async로 맞춘다(실모델 백엔드와 같은 인터페이스 — 노드는 모두 await로 부름).
[개선 I-4] judge(): 결정론적 심판. 실제 LLM 심판의 4개 기준을 규칙으로 흉내 낸다 — 수집된 매뉴얼(특히 안전 이슈)이
답변에 빠졌거나(Completeness), 안전 이슈를 뒤에 다뤘거나(Domain), 조사한 설비를 언급하지 않으면(Relevance) 불합격.
QualityLLM.write는 '짧게/간결하게' 요청을 그대로 따르는 초안을 쓴다 — 실제 LLM이 사용자 지시를 따르다 안전 조치를
빼먹는 상황을 재현해, 코드 검사만으로는 통과하지만 심판이 잡아내는 경우를 결정론적으로 보여 주기 위함.
"""
import json
import re

from langsmith import traceable

from .config import RULES, incident_types_by_priority
from .diagnosis import evidence_for
from .equipment import DEFAULT_EQUIPMENT, REFERENCE_EQUIPMENT, describe, find_mentions

# [Base 대비 변경] IT 장애 단어(장애|진단|다운|지연 …) → 분산에너지 이상 단어
# [After 변경 — 모듈화] 단어 목록을 config/rules.json의 route_keywords에서 읽는다
INCIDENT_PATTERN = RULES["route_keywords"]["incident_pattern"]
BREVITY_PATTERN = r"짧게|간결|\d+\s*자\s*이내|한 줄로"
JUDGE_FEEDBACK_TAG = "[심판]"


def parse_judgement(content: str) -> dict:
    """심판 LLM의 JSON 응답을 {is_acceptable, reasoning}으로. 형식이 깨지면 통과로 본다(검수 실패로 답이 막히지 않게)."""
    try:
        raw = json.loads(content)
        return {"is_acceptable": bool(raw.get("is_acceptable", True)),
                "reasoning": str(raw.get("reasoning", "")).strip()}
    except (json.JSONDecodeError, AttributeError):
        return {"is_acceptable": True, "reasoning": "심판 응답 형식 오류 — 통과 처리"}


def runbook_short_title(text: str) -> str:
    """'# Runbook: ESS 셀 과열(Overheat) 대응 절차' → 'ESS 셀 과열(Overheat)'"""
    title, _ = _runbook_title_steps(text)
    return re.sub(r"^Runbook:\s*", "", title).split(" 대응")[0].strip()


class FakeLLM:
    def __init__(self, name: str, usd_per_1k_tokens: float):
        self.name = name
        self.usd_per_1k = usd_per_1k_tokens

    def _usage(self, prompt: str, completion: str) -> dict:
        tokens = (len(prompt) + len(completion)) // 4 + 1
        return {"model": self.name, "tokens": tokens,
                "usd": round(tokens / 1000 * self.usd_per_1k, 6)}


class FastLLM(FakeLLM):
    """분류·계획 등 짧고 정형적인 단계 전담 (저비용)."""

    @traceable(run_type="llm", name="fast_llm.route")
    async def route(self, query: str) -> tuple[dict, dict]:
        """Wave 5 — 쿼리 라우팅. 설비 이상 관련이면 incident 파이프라인, 아니면 chat 직행."""
        incident = bool(re.search(INCIDENT_PATTERN, query))
        result = {"route": "incident" if incident else "chat", "needs_analysis": False}
        return result, self._usage(query, str(result))

    @traceable(run_type="llm", name="fast_llm.plan")
    async def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        """Wave 4 — Plan/Task 분해. MCP Prompt가 준 표준 절차를 따라 계획을 세운다.

        Wave 6 재현: needs_analysis=False라고 스스로 판단해 놓고도 analysis task를
        끼워 넣는 '모델의 자기모순'을 결정론적으로 재현한다 → gate 노드가 걸러낸다.
        """
        # [Base 대비 변경] 영어 서버 이름 정규식 → 설비 ID + 한국어 별칭 찾기
        systems = find_mentions(query) or [DEFAULT_EQUIPMENT]
        if REFERENCE_EQUIPMENT not in systems:
            systems.append(REFERENCE_EQUIPMENT)   # 비교용 단지 부하 조회 → Wave 10b 병렬 실행 대상
        tasks = [
            {"id": f"task-{i}", "agent": "research", "kind": "monitoring",
             "target": s, "status": "pending"}
            for i, s in enumerate(dict.fromkeys(systems))
        ]
        tasks.append({"id": "task-analysis", "agent": "analysis",
                      "kind": "pattern-analysis", "status": "pending"})   # 자기모순 task
        result = {"needs_analysis": False, "tasks": tasks}
        return result, self._usage(query + procedure_prompt, str(result))

    @traceable(run_type="llm", name="fast_llm.analyze")
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        """Wave 6 보완 — 여러 설비의 계측 결과 사이에 단계 간 연쇄가 있는지 비교."""
        mon = [c["data"] for c in collected if c["source"] == "monitoring"]
        degraded = [d["equipment_id"] for d in mon if d.get("status") != "healthy"]
        stages = sorted({d.get("stage", "") for d in mon if d.get("status") != "healthy"})
        if len(degraded) >= 2:
            summary = (f"{', '.join(degraded)} 복수 설비가 동시에 이상 "
                       f"({'·'.join(stages)} 단계) — 단계 간 연쇄 영향이 의심됨")
        elif degraded:
            summary = f"{degraded[0]}만 단독 이상 — 연관 설비는 정상, 국소 이슈로 추정"
        else:
            summary = "조회한 설비 모두 healthy — 이상 징후 없음"
        return summary, self._usage(str(collected), summary)

    @traceable(run_type="llm", name="fast_llm.judge")
    async def judge(self, query: str, collected: list[dict], draft: str) -> tuple[dict, dict]:
        """[개선 I-4] 결정론적 심판 — Relevance / Completeness / Domain을 규칙으로 판정 (Clarity는 코드 검사가 담당)."""
        problems = []
        mon_ids = [c["data"]["equipment_id"] for c in collected if c["source"] == "monitoring"]
        if mon_ids and not any(eq in draft for eq in mon_ids):
            problems.append("Relevance: 조사한 설비가 답변에 언급되지 않았습니다")
        runs = [c for c in collected if c["source"] == "runbook"]
        for c in runs:
            short = runbook_short_title(c["data"])
            if short not in draft:
                label = "안전 이슈 매뉴얼" if c.get("role") == "safety" else "대응 매뉴얼"
                problems.append(f"Completeness: {label} '{short}'의 조치가 빠졌습니다")
        # Domain: 가장 위험한(우선순위가 높은) 안전 이슈의 조치가 다른 조치보다 먼저 나와야 한다
        ordered = [runbook_short_title(c["data"]) for c in _by_priority(runs)]
        if ordered and ordered[0] in draft and any(
                t in draft and draft.index(t) < draft.index(ordered[0]) for t in ordered[1:]):
            problems.append("Domain: 안전 이슈를 다른 조치보다 먼저 다뤄야 합니다")
        result = {"is_acceptable": not problems,
                  "reasoning": " / ".join(problems) or "4개 기준 모두 충족"}
        return result, self._usage(query + draft, result["reasoning"])

    @traceable(run_type="llm", name="fast_llm.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        """Wave 5의 direct 경로 + Wave 12 증명용 — 히스토리에서 이름을 기억해낸다.

        fake 백엔드는 실제 네트워크 호출이 없어 토큰 단위로 쪼갤 대상이 없다 —
        ollama_llm.py와 인터페이스(on_token 콜백)만 맞추고 결과를 한 번에 반환한다."""
        for msg in reversed(history):
            m = re.search(r"제 이름은 (\S+?)입니다", msg.get("content", ""))
            if m and re.search(r"이름이 뭐", query):
                reply = f"{m.group(1)}님이라고 말씀하셨어요. 무엇을 도와드릴까요?"
                if on_token is not None:
                    on_token(reply)
                return reply, self._usage(query, reply)
        m = re.search(r"제 이름은 (\S+?)입니다", query)
        if m:
            reply = f"안녕하세요, {m.group(1)}님! 무엇을 도와드릴까요?"
        else:
            reply = "안녕하세요! 무엇을 도와드릴까요?"
        if on_token is not None:
            on_token(reply)
        return reply, self._usage(query, reply)


class QualityLLM(FakeLLM):
    """최종 답변 작성 전담 (고품질·고비용)."""

    @traceable(run_type="llm", name="quality_llm.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
        """Wave 8 재현: 1차 초안은 의도적으로 저품질(너무 짧음) → validator가 반려하면
        피드백을 반영한 구조화된 2차 초안을 내놓는다.

        fake 백엔드는 실제 네트워크 호출이 없어 토큰 단위로 쪼갤 대상이 없다 —
        ollama_llm.py와 인터페이스(on_token 콜백)만 맞추고 결과를 한 번에 반환한다."""
        if attempt == 1:
            draft = "재시작하면 됩니다."
            if on_token is not None:
                on_token(draft)
            return draft, self._usage(query, draft)

        # [After 변경] 전문 에이전트 점검 결과와 매뉴얼 여러 개(주 매뉴얼 + 안전 매뉴얼)를 반영
        mon = [c for c in collected if c["source"] == "monitoring"]
        runs = [c for c in collected if c["source"] == "runbook"]
        checks = [chk for c in collected if c["source"] == "specialist" for chk in c["data"]["checks"]]

        # [개선 I-4] 사용자가 '짧게'를 요구하면 그대로 따른다(심판에게 반려되기 전까지) — 실모델의 흔한 실수 재현
        if re.search(BREVITY_PATTERN, query) and JUDGE_FEEDBACK_TAG not in (feedback or ""):
            primary = next((c for c in runs if c.get("role", "primary") != "safety"), None)
            step = _runbook_title_steps(primary["data"])[1][1:2] if primary else []
            draft = f"1) {checks[0] if checks else '점검 결과 확인 필요'} 2) {step[0] if step else '표준 절차대로 조치'}"
            if on_token is not None:
                on_token(draft)
            return draft, self._usage(query + str(feedback), draft)
        lines = [f"'{query}' 진단 결과:",
                 f"1) 현재 상태: {' · '.join(describe(c['data']) for c in mon)}",
                 f"2) 원인 추정: {'; '.join(checks) if checks else '전문 에이전트 점검에서 기준 초과 항목 없음'}"]
        n = 3
        for c in _by_priority(runs):   # 우선순위가 높은(더 위험한) 이상의 조치부터
            title, steps = _runbook_title_steps(c["data"])
            safety = RULES["incident_types"].get(c.get("incident_type", ""), {}).get("safety")
            label = "우선 안전 조치" if safety or c.get("role") == "safety" else "권장 조치"
            situation, sit_steps = _situation_steps(c["data"])
            if c.get("unconfirmed"):   # 질문한 이상이 데이터로 확인되지 않음 → 설비를 움직이는 조치는 빼고 확인 절차만
                ev = "; ".join(evidence_for(collected, c.get("incident_type", ""))) or "관련 계측 항목 없음"
                checks_only = [st for st in (sit_steps or steps) if not _is_control(st)]
                lines.append(f"{n}) 확인 필요 — 데이터상 이상이 확인되지 않음('{title}'): {ev}. "
                             f"설비를 멈추기 전에 현장·센서를 먼저 확인하세요: {' '.join(checks_only)}")
                n += 1
                continue
            if situation:   # 매뉴얼의 '해당 상황' 절 — 그 상황의 조치는 하나도 빼지 않는다(화재 시 대피 등)
                advice = " ".join(sit_steps)
                if steps[1:3] and steps[1] not in sit_steps:   # 기본 절차도 따르는 상황이면 핵심 2단계를 앞에
                    advice = " ".join(steps[1:3]) + " " + advice
                label += f", 상황: {situation}"
            else:
                advice = " ".join(steps[1:3]) or "표준 절차를 따라 조치하세요."
            lines.append(f"{n}) {label}('{title}'): {advice}")
            n += 1
        for c in collected:   # 운영자가 승인 단계에서 남긴 판단 — 보고서에 반영
            if c["source"] == "operator":
                lines.append(f"{n}) 운영자 판단 반영: {c['data']}")
                n += 1
        failed = [f"{c['data']['target']}" for c in collected if c["source"] == "failure"]
        if failed:   # [개선 I-1] 데이터를 못 받은 설비는 추정하지 않고 '확인 불가'로 명시
            lines.append(f"{n}) 확인 불가: {', '.join(failed)} 조회에 실패해 판단에서 제외했습니다 — 현장 확인이 필요합니다.")
            n += 1
        lines.append(f"{n}) 정상화 확인 후 원인 분석과 재발 방지 대책을 기록하세요.")
        draft = " ".join(lines)
        if on_token is not None:
            on_token(draft)
        return draft, self._usage(query + str(feedback), draft)


def _is_control(step: str) -> bool:
    cfg = RULES["approval"]
    return cfg["marker"] in step or any(p in step for p in cfg["action_phrases"])


def _by_priority(runs: list[dict]) -> list[dict]:
    """매뉴얼 발췌들을 이상 유형 우선순위 순서로(안전 > 데이터 신뢰 > 계통 > 비용 > 효율). 유형이 없으면 안전 역할 먼저."""
    order = incident_types_by_priority()
    rank = lambda c: (order.index(c["incident_type"]) if c.get("incident_type") in order
                      else (-1 if c.get("role") == "safety" else len(order)))
    return sorted(runs, key=rank)


def _situation_steps(text: str) -> tuple[str | None, list[str]]:
    """발췌에 '## 해당 상황: …' 절이 있으면 (상황 이름, 그 절의 번호 절차)."""
    if "## 해당 상황:" not in text:
        return None, []
    head, body = text.split("## 해당 상황:", 1)
    lines = body.strip().splitlines()
    steps = [m.group(1) for l in lines[1:] if (m := re.match(r"^\d+\.\s+(.*)", l.strip()))]
    return lines[0].strip(), steps


def _runbook_title_steps(text: str) -> tuple[str, list[str]]:
    text = text.split("## 해당 상황:", 1)[0]   # 기본 부분의 절차만(상황 절차는 _situation_steps)
    body = [l.strip() for l in text.strip().splitlines() if l.strip()]
    title = body[0].lstrip("# ")
    steps = [m.group(1) for l in body[1:] if (m := re.match(r"^\d+\.\s+(.*)", l))]
    return title, steps
