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

[개선 I-2] 동기 httpx.post → httpx.AsyncClient + await (이벤트 루프를 막지 않음, ollama_llm.py와 같은 이유).
[개선 I-11] 동시 호출 수는 agent/limits.py의 llm_slot()으로 제한.
[개선 I-4] judge(): 작성 모델(gpt-4o)과 별개인 fast 모델(gpt-4o-mini)이 보고서를 4개 기준으로 검수.
"""
import json
import os
import re

import httpx
from langsmith import traceable

from .equipment import DEFAULT_EQUIPMENT, KNOWN_SYSTEMS as EQUIPMENT_IDS, find_mentions
from .fake_llm import INCIDENT_PATTERN, parse_judgement
from .limits import llm_slot
from .prompts import (ANALYZE_SYSTEM, CHAT_SYSTEM, JUDGE_SYSTEM, ROUTE_SYSTEM, WRITE_SYSTEM,
                      analyze_facts, judge_user, operator_rule, plan_system, write_facts)

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
    '{"route": "chat" 또는 "equipment" 또는 "incident", "needs_analysis": true 또는 false}'
)

PLAN_INSTRUCTIONS = (
    'JSON 객체로만 응답하라. 스키마: {"needs_analysis": boolean, '
    '"tasks": [{"agent": "research"|"analysis", "target": string(선택)}]}'
)


class OpenAILLM:
    def __init__(self, model: str, usd_per_1k_tokens: float, judge_model: str | None = None,
                 judge_usd_per_1k: float | None = None):
        self.name = f"openai:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        # [개선 I-4 보정] 심판 모델을 따로 정할 수 있다. OpenAI 실측에서 gpt-4o-mini 심판이 안전 이슈가 없는 질문도
        # 3회 연속 반려(자기모순적 이유)해 재작성 비용이 커졌고, gpt-4o 심판은 같은 질문을 1회에 통과시켰다.
        self.judge_model = judge_model or model
        self.judge_usd_per_1k = judge_usd_per_1k if judge_usd_per_1k is not None else usd_per_1k_tokens

    async def _chat(self, system: str, user: str, json_mode: bool = False,
                    model: str | None = None, usd_per_1k: float | None = None) -> tuple[str, dict]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다 — .env 또는 환경변수로 지정하세요."
            )
        payload = {
            "model": model or self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        async with llm_slot(), httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OPENAI_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        price = self.usd_per_1k if usd_per_1k is None else usd_per_1k
        usage = {"model": f"openai:{model or self.model}", "tokens": tokens,
                 "usd": round(tokens / 1000 * price, 6)}
        return content, usage


class OpenAIFastLLM(OpenAILLM):
    """[분산에너지] 분류·계획·단순 대화 전담 (저비용 모델, 기본 gpt-4o-mini)."""

    @traceable(run_type="llm", name="openai_fast.route")
    async def route(self, query: str) -> tuple[dict, dict]:
        system = ROUTE_SYSTEM + " " + ROUTE_INSTRUCTIONS   # [After] prompts.py로 이동
        content, usage = await self._chat(system, query, json_mode=True)
        try:
            raw = json.loads(content)
            route = raw.get("route") if raw.get("route") in ("chat", "equipment", "incident") else None
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
    async def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        system = plan_system(sorted(KNOWN_SYSTEMS)) + " " + PLAN_INSTRUCTIONS   # [After] prompts.py로 이동
        user = f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"
        content, usage = await self._chat(system, user, json_mode=True)

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
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        system = ANALYZE_SYSTEM   # [After] prompts.py로 이동
        facts = analyze_facts(collected)
        return await self._chat(system, facts)

    @traceable(run_type="llm", name="openai_fast.judge")
    async def judge(self, query: str, collected: list[dict], draft: str) -> tuple[dict, dict]:
        content, usage = await self._chat(JUDGE_SYSTEM, judge_user(query, collected, draft), json_mode=True,
                                          model=self.judge_model, usd_per_1k=self.judge_usd_per_1k)
        return parse_judgement(content), usage

    @traceable(run_type="llm", name="openai_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        """ollama_llm.py와 인터페이스(on_token 콜백)만 맞춘다 — OpenAI Chat Completions를
        아직 stream=True로 호출하지 않아 토큰 단위 중계는 하지 않고 완성된 응답을 한 번에
        전달한다."""
        system = CHAT_SYSTEM   # [After] prompts.py로 이동
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
        user = (f"## 지금까지의 대화\n{rendered or '(없음)'}\n\n"
                f"## 사용자의 새 메시지\n{query}")
        content, usage = await self._chat(system, user)
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
        system = WRITE_SYSTEM   # [After] prompts.py로 이동
        user = f"## 사용자 요청\n{query}\n\n## 수집된 사실\n" + write_facts(collected) + operator_rule(collected)
        if feedback:
            user += f"\n\n## 이전 초안에 대한 검증 반려 사유 (반드시 반영)\n{feedback}"
        content, usage = await self._chat(system, user)
        usage["prompt"] = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        if on_token is not None:
            on_token(content)
        return content, usage
