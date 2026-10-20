"""LLM_BACKEND=ollama — 로컬 Ollama 실모델 백엔드 (fake_llm과 동일 인터페이스).

Wave 10a 구도를 실물로: fast(기본 qwen3:0.6b)는 분류·계획·단순 대화, quality(기본
qwen3:8b)는 최종 답변 작성 전담. Ollama의 구조화 출력(format=<JSON Schema>)과
temperature=0으로 변동을 줄이지만 실모델이므로 결정론은 아니다 — 그래서 gate(W6)·
replan(W7)·validator(W8)가 연출이 아니라 실제 방어선으로 동작한다.
fc
비용(cost_usd)은 '가상 단가 × 실측 토큰 수(prompt_eval_count+eval_count)'다 — 로컬
추론이라 실제 과금은 0이지만, W10a의 모델 분리 효과를 같은 잣대로 비교하기 위함이다.

사전 조건: `ollama serve` 실행 중 + `ollama pull qwen3:0.6b qwen3:8b`.
"""
import json
import re

import httpx
from langsmith import traceable

from .equipment import DEFAULT_EQUIPMENT, KNOWN_SYSTEMS as EQUIPMENT_IDS, describe, find_mentions
from .fake_llm import INCIDENT_PATTERN

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


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


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["chat", "incident"]},
        "needs_analysis": {"type": "boolean"},
    },
    "required": ["route", "needs_analysis"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_analysis": {"type": "boolean"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string",
                              "enum": ["research", "analysis"]},
                    "target": {"type": "string"},
                },
                "required": ["agent"],
            },
        },
    },
    "required": ["needs_analysis", "tasks"],
}


class OllamaLLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"ollama:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens

    def _chat(self, system: str, user: str, schema: dict | None = None) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if schema is not None:
            payload["format"] = schema
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        usage = {"model": self.name, "tokens": tokens,
                 "usd": round(tokens / 1000 * self.usd_per_1k, 6)}
        return content, usage

    async def _chat_stream(self, system: str, user: str, on_token=None) -> tuple[str, dict]:
        """사용자 대면 텍스트(최종 답변류) 전용 — Ollama NDJSON 스트림을 토큰 단위로
        받아 on_token(조각)을 즉시 호출해 SSE로 실시간 중계할 수 있게 한다. 분류/계획
        등 JSON 스키마 출력(route/plan/analyze)은 부분 출력이 의미가 없어 그대로 _chat을
        쓴다."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "options": {"temperature": 0.0},
        }
        chunks: list[str] = []
        usage = {"model": self.name, "tokens": 0, "usd": 0.0}
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        chunks.append(piece)
                        if on_token is not None:
                            on_token(piece)
                    if data.get("done"):
                        tokens = ((data.get("prompt_eval_count") or 0)
                                  + (data.get("eval_count") or 0))
                        usage = {"model": self.name, "tokens": tokens,
                                 "usd": round(tokens / 1000 * self.usd_per_1k, 6)}
        usage["prompt"] = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        return "".join(chunks), usage


class OllamaFastLLM(OllamaLLM):
    """[분산에너지] 분류·계획·단순 대화 전담 (저사양 소형 모델)."""

    @traceable(run_type="llm", name="ollama_fast.route")
    def route(self, query: str) -> tuple[dict, dict]:
        system = ("당신은 분산에너지(태양광·ESS·연계점·EV충전기 등) 운영 지원 라우터다. "
                  "사용자 요청을 분류하라. 설비 이상·출력 저하·피크·전압·충전율·과열·"
                  "통신 두절·출력제어·수요반응(DR) 등 운영 이슈면 route=incident, "
                  "인사·잡담·일반 질문이면 route=chat. "
                  "needs_analysis는 패턴 비교/통계 분석이 꼭 필요할 때만 true.")
        content, usage = self._chat(system, query, schema=ROUTE_SCHEMA)
        try:
            raw = json.loads(content)
            route = raw.get("route") if raw.get("route") in ("chat", "incident") else None
            result = {"route": route or "chat",
                      "needs_analysis": bool(raw.get("needs_analysis", False))}
        except (json.JSONDecodeError, AttributeError):
            # 소형 모델의 형식 붕괴 대비 — 코드 레벨 fallback (W6 정신)
            incident = bool(re.search(INCIDENT_PATTERN, query))
            result = {"route": "incident" if incident else "chat",
                      "needs_analysis": False}
        return result, usage

    @traceable(run_type="llm", name="ollama_fast.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        system = ("당신은 분산에너지 운영 Supervisor의 planning 노드다. 표준 진단 절차에 따라 "
                  "실행 계획(tasks)을 JSON으로 세워라. "
                  "사용자 요청에 등장한 설비마다 agent=research task를 하나씩 세우고, "
                  "target에는 반드시 설비 ID를 넣어라 "
                  f"(알려진 설비 ID: {', '.join(sorted(KNOWN_SYSTEMS))} — 예: 태양광 2호기=pv-02, "
                  "ESS 1호기=ess-01, 변압기=tr-01, 연계점=pcc-01. 특정 설비가 언급되지 "
                  "않았으면 pcc-01을 기본으로 조사하라). "
                  "패턴 분석이 꼭 필요하다고 판단되면 agent=analysis task를 추가해도 된다.")
        user = f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"
        content, usage = self._chat(system, user, schema=PLAN_SCHEMA)

        # 소형 모델 출력에 대한 코드 레벨 정규화 — 스키마는 강제돼도 내용은 흔들릴 수
        # 있으므로(대상 시스템 오타, task 누락/중복) 실행 가능한 형태로 보정한다.
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

    @traceable(run_type="llm", name="ollama_fast.analyze")
    def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        system = ("당신은 분산에너지 운영 분석가다. 여러 설비의 계측 결과를 비교해 "
                  "생산·이동·저장·공급 단계 사이의 연쇄 영향이나 공통 원인이 있는지 "
                  "한두 문장으로 짧게 답하라. "
                  "패턴이 없으면 '특이 패턴 없음'이라고만 답하라.")
        mon = [c["data"] for c in collected if c["source"] == "monitoring"]
        facts = "\n".join(f"- {describe(d)}" for d in mon) or "(계측 데이터 없음)"
        return self._chat(system, facts)

    @traceable(run_type="llm", name="ollama_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        system = ("당신은 분산에너지 단지 운영을 돕는 간결한 한국어 비서다. 한두 문장으로 답하라. "
                  "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라.")
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
        user = (f"## 지금까지의 대화\n{rendered or '(없음)'}\n\n"
                f"## 사용자의 새 메시지\n{query}")
        return await self._chat_stream(system, user, on_token=on_token)


class OllamaQualityLLM(OllamaLLM):
    """최종 답변 작성 전담 (상대적으로 큰 모델)."""

    @traceable(run_type="llm", name="ollama_quality.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
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
        return await self._chat_stream(system, user, on_token=on_token)
