"""LLM_BACKEND=openai — 실제 OpenAI API 백엔드 (fake_llm/ollama_llm과 동일 인터페이스).

Wave 10a 구도를 실제 상용 API로: fast(기본 gpt-4o-mini)는 분류·계획·단순 대화,
quality(기본 gpt-4o)는 최종 답변 작성 전담. ollama_llm.py와 마찬가지로 실모델이라
결정론은 아니다 — gate(W6)·replan(W7)·validator(W8)가 연출이 아니라 실제 방어선으로
동작한다. Ollama와 달리 실제 과금이 발생하는 원격 API이므로, 이 백엔드로 실행하면
OpenAI 계정에 실제 비용이 청구된다(run_openai.sh 실행 전 반드시 인지할 것).

구조화 출력은 OpenAI Chat Completions의 `response_format={"type": "json_object"}`
(JSON 모드)로 강제한다 — Ollama의 `format=<JSON Schema>`처럼 스키마 자체를 강제하는
기능은 아니라서, 시스템 프롬프트에 원하는 필드를 명시하고 ollama_llm.py와 동일하게
코드 레벨 방어(파싱 실패·필드 누락 시 보정)를 그대로 둔다.

비용(cost_usd)은 OpenAI 응답의 `usage.total_tokens`(실측) × 가상 단가다 — 실제 청구
금액과는 별개로, W10a의 모델 분리 효과를 fake/ollama와 같은 잣대로 비교하기 위함이다.
(OpenAI는 입력/출력 토큰 단가가 실제로는 다르지만, 세 백엔드의 인터페이스를
`usd_per_1k_tokens` 하나로 통일해두기 위해 여기서도 단일 블렌디드 단가로 단순화한다.)

사전 조건: 유효한 `OPENAI_API_KEY` 환경변수(.env).
"""
import json
import os
import re

import httpx
from langsmith import traceable

from .equipment import DEFAULT_EQUIPMENT, KNOWN_SYSTEMS as EQUIPMENT_IDS, describe, find_mentions
from .fake_llm import INCIDENT_PATTERN

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# [Base 대비 변경] monitoring.json의 서버 이름을 파일마다 따로 읽던 코드 → 설비 정보를
# agent/equipment.py 한 곳에서 가져온다(설비 ID + 한국어 별칭 찾기 포함).
KNOWN_SYSTEMS = EQUIPMENT_IDS


def _guess_target(query: str) -> str:
    """plan()의 코드 레벨 정규화에서 쓰는 안전망 — 모델이 target을 아예 안 내놓거나
    오타를 내놓으면 질문에서 설비 ID·별칭을 다시 찾아 보정한다(W6 정신: 모델 출력을
    그대로 믿지 않고 코드가 보정). 아무것도 안 걸리면 연계점(pcc-01)으로 폴백."""
    mentions = find_mentions(query)
    return mentions[0] if mentions else DEFAULT_EQUIPMENT


def _guess_targets(query: str) -> list[str]:
    """research task가 통째로 0개일 때 쓰는 복구용 — 질문에 등장하는 설비 전부를 복구한다
    (하나만 고르면 나머지 설비가 조사에서 빠지고, write 단계가 빈 자리의 수치를 지어낼 수 있음)."""
    return find_mentions(query) or [DEFAULT_EQUIPMENT]


ROUTE_INSTRUCTIONS = (
    'JSON 객체로만 응답하라. 스키마: '
    '{"route": "chat" 또는 "incident", "needs_analysis": true 또는 false}'
)

PLAN_INSTRUCTIONS = (
    'JSON 객체로만 응답하라. 스키마: {"needs_analysis": boolean, '
    '"tasks": [{"agent": "research"|"analysis", "target": string(선택)}]}'
)


class OpenAILLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"openai:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def _chat(self, system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다 — .env 또는 환경변수로 지정하세요."
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        resp = httpx.post(OPENAI_URL, json=payload, headers=headers, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        usage = {"model": self.name, "tokens": tokens,
                 "usd": round(tokens / 1000 * self.usd_per_1k, 6)}
        return content, usage


class OpenAIFastLLM(OpenAILLM):
    """[분산에너지] 분류·계획·단순 대화 전담 (저비용 모델, 기본 gpt-4o-mini)."""

    @traceable(run_type="llm", name="openai_fast.route")
    def route(self, query: str) -> tuple[dict, dict]:
        system = ("당신은 분산에너지(태양광·ESS·연계점·EV충전기 등) 운영 지원 라우터다. "
                  "사용자 요청을 분류하라. 설비 이상·출력 저하·피크·전압·충전율·과열·"
                  "통신 두절·출력제어·수요반응(DR) 등 운영 이슈면 route=incident, "
                  "인사·잡담·일반 질문이면 route=chat. "
                  "needs_analysis는 패턴 비교/통계 분석이 꼭 필요할 때만 true. "
                  + ROUTE_INSTRUCTIONS)
        content, usage = self._chat(system, query, json_mode=True)
        try:
            raw = json.loads(content)
            route = raw.get("route") if raw.get("route") in ("chat", "incident") else None
            result = {"route": route or "chat",
                      "needs_analysis": bool(raw.get("needs_analysis", False))}
        except (json.JSONDecodeError, AttributeError):
            # JSON 모드는 "유효한 JSON"만 보장할 뿐 스키마까지 강제하진 않으므로,
            # 필드 누락·오탐 시 코드 레벨 fallback을 둔다(W6 정신, ollama_llm.py와 동일).
            incident = bool(re.search(INCIDENT_PATTERN, query))
            result = {"route": "incident" if incident else "chat",
                      "needs_analysis": False}
        return result, usage

    @traceable(run_type="llm", name="openai_fast.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        system = ("당신은 분산에너지 운영 Supervisor의 planning 노드다. 표준 진단 절차에 따라 "
                  "실행 계획(tasks)을 JSON으로 세워라. "
                  "사용자 요청에 등장한 설비마다 agent=research task를 하나씩 세우고, "
                  "target에는 반드시 설비 ID를 넣어라 "
                  f"(알려진 설비 ID: {', '.join(sorted(KNOWN_SYSTEMS))} — 예: 태양광 2호기=pv-02, "
                  "ESS 1호기=ess-01, 변압기=tr-01, 연계점=pcc-01. 특정 설비가 언급되지 "
                  "않았으면 pcc-01을 기본으로 조사하라). "
                  "패턴 분석이 꼭 필요하다고 판단되면 agent=analysis task를 추가해도 된다. "
                  + PLAN_INSTRUCTIONS)
        user = f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"
        content, usage = self._chat(system, user, json_mode=True)

        # 코드 레벨 정규화 — JSON 형식은 지켜져도 내용은 흔들릴 수 있으므로
        # (대상 시스템 오타, task 누락/중복) 실행 가능한 형태로 보정한다.
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            raw = {"needs_analysis": False, "tasks": []}
        tasks, seen, idx = [], set(), 0
        analysis_added = False
        for t in raw.get("tasks", []):
            agent = t.get("agent")
            if agent == "research":
                target = t.get("target", "")
                target = target if target in KNOWN_SYSTEMS else _guess_target(query)
                if target in seen:
                    continue
                seen.add(target)
                tasks.append({"id": f"task-{idx}", "agent": "research",
                              "kind": "monitoring", "target": target,
                              "status": "pending"})
                idx += 1
            elif agent == "analysis" and not analysis_added:
                analysis_added = True
                tasks.append({"id": "task-analysis", "agent": "analysis",
                              "kind": "pattern-analysis", "status": "pending"})
        if not seen:   # research가 하나도 없으면 실행 불능 — 쿼리에 등장하는
            # KNOWN_SYSTEMS 전부를 복구(하나만 복구하면 나머지가 조사 누락된 채
            # write 단계에서 수치가 조작될 수 있음)
            recovered = [{"id": f"task-{i}", "agent": "research",
                          "kind": "monitoring", "target": target,
                          "status": "pending"}
                         for i, target in enumerate(_guess_targets(query))]
            tasks = recovered + tasks
        result = {"needs_analysis": bool(raw.get("needs_analysis", False)),
                  "tasks": tasks}
        return result, usage

    @traceable(run_type="llm", name="openai_fast.analyze")
    def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        system = ("당신은 분산에너지 운영 분석가다. 여러 설비의 계측 결과를 비교해 "
                  "생산·이동·저장·공급 단계 사이의 연쇄 영향이나 공통 원인이 있는지 "
                  "한두 문장으로 짧게 답하라. "
                  "패턴이 없으면 '특이 패턴 없음'이라고만 답하라.")
        mon = [c["data"] for c in collected if c["source"] == "monitoring"]
        facts = "\n".join(f"- {describe(d)}" for d in mon) or "(계측 데이터 없음)"
        return self._chat(system, facts)

    @traceable(run_type="llm", name="openai_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        """ollama_llm.py와 인터페이스(on_token 콜백)만 맞춘다 — OpenAI Chat Completions를
        아직 stream=True로 호출하지 않아 토큰 단위 중계는 하지 않고 완성된 응답을 한 번에
        전달한다."""
        system = ("당신은 분산에너지 단지 운영을 돕는 간결한 한국어 비서다. 한두 문장으로 답하라. "
                  "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라.")
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
        user = (f"## 지금까지의 대화\n{rendered or '(없음)'}\n\n"
                f"## 사용자의 새 메시지\n{query}")
        content, usage = self._chat(system, user)
        usage["prompt"] = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        if on_token is not None:
            on_token(content)
        return content, usage


class OpenAIQualityLLM(OpenAILLM):
    """최종 답변 작성 전담 (고품질 모델, 기본 gpt-4o)."""

    @traceable(run_type="llm", name="openai_quality.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
        """ollama_llm.py와 인터페이스(on_token 콜백)만 맞춘다 — OpenAI Chat Completions를
        아직 stream=True로 호출하지 않아 토큰 단위 중계는 하지 않고 완성된 응답을 한 번에
        전달한다."""
        system = ("당신은 분산에너지 운영 보고 작성자다. 수집된 사실만 사용해 한국어로 "
                  "간결한 대응 보고를 작성하라. 반드시 '1) 현재 상태 2) 원인 추정 "
                  "3) 권장 조치' 형식의 번호로 답하고, 수치는 수집된 값을 단위까지 그대로 "
                  "인용하라. 설비를 움직이는 조치(충방전·출력 제한·부하 차단 등)에는 "
                  "'(운영자 승인 필요)'를 붙여라.")
        facts = []
        for c in collected:
            if c["source"] == "monitoring":
                facts.append(f"- 설비 계측 {describe(c['data'])}")
            elif c["source"] == "runbook":
                facts.append(f"- 대응 매뉴얼 발췌:\n{c['data'].strip()}")
            elif c["source"] == "analysis":
                facts.append(f"- 연쇄 분석: {c['data'].strip()}")
        user = f"## 사용자 요청\n{query}\n\n## 수집된 사실\n" + "\n".join(facts)
        if feedback:
            user += f"\n\n## 이전 초안에 대한 검증 반려 사유 (반드시 반영)\n{feedback}"
        content, usage = self._chat(system, user)
        usage["prompt"] = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        if on_token is not None:
            on_token(content)
        return content, usage
