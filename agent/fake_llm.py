"""Wave 4/10a — 결정론적 Fake LLM 2종 (fast/quality) + 실측 비용 계산.

실제 LLM 호출 없이도 파이프라인 전체가 결정론적으로 재현되도록, 프롬프트 내용에 따라
정해진 응답을 돌려준다. 각 호출은 LangSmith에 run_type="llm"으로 기록되고, 문자열
길이 기반 토큰 추정치로 비용을 계산해 상태에 누적한다 (Wave 10a의 모델 분리 근거).
"""
import re

from langsmith import traceable

from .config import RULES
from .equipment import DEFAULT_EQUIPMENT, REFERENCE_EQUIPMENT, describe, find_mentions

# [Base 대비 변경] IT 장애 단어(장애|진단|다운|지연 …) → 분산에너지 이상 단어
# [After 변경 — 모듈화] 단어 목록을 config/rules.json의 route_keywords에서 읽는다
INCIDENT_PATTERN = RULES["route_keywords"]["incident_pattern"]


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
    def route(self, query: str) -> tuple[dict, dict]:
        """Wave 5 — 쿼리 라우팅. 설비 이상 관련이면 incident 파이프라인, 아니면 chat 직행."""
        incident = bool(re.search(INCIDENT_PATTERN, query))
        result = {"route": "incident" if incident else "chat", "needs_analysis": False}
        return result, self._usage(query, str(result))

    @traceable(run_type="llm", name="fast_llm.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
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
    def analyze(self, collected: list[dict]) -> tuple[str, dict]:
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
        lines = [f"'{query}' 진단 결과:",
                 f"1) 현재 상태: {' · '.join(describe(c['data']) for c in mon)}",
                 f"2) 원인 추정: {'; '.join(checks) if checks else '전문 에이전트 점검에서 기준 초과 항목 없음'}"]
        n = 3
        ordered = ([c for c in runs if c.get("role") == "safety"] +
                   [c for c in runs if c.get("role", "primary") != "safety"])
        for c in ordered:
            title, steps = _runbook_title_steps(c["data"])
            label = "우선 안전 조치" if c.get("role") == "safety" else "권장 조치"
            advice = " ".join(steps[1:3]) or "표준 절차를 따라 조치하세요."
            lines.append(f"{n}) {label}('{title}'): {advice}")
            n += 1
        lines.append(f"{n}) 정상화 확인 후 원인 분석과 재발 방지 대책을 기록하세요.")
        draft = " ".join(lines)
        if on_token is not None:
            on_token(draft)
        return draft, self._usage(query + str(feedback), draft)


def _runbook_title_steps(text: str) -> tuple[str, list[str]]:
    body = [l.strip() for l in text.strip().splitlines() if l.strip()]
    title = body[0].lstrip("# ")
    steps = [m.group(1) for l in body[1:] if (m := re.match(r"^\d+\.\s+(.*)", l))]
    return title, steps
