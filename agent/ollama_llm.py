"""LLM_BACKEND=ollama — 로컬 Ollama 실모델 백엔드 (fake_llm과 동일 인터페이스).

Wave 10a 구도를 실물로: fast(기본 qwen3:0.6b)는 분류·계획·단순 대화, quality(기본
qwen3:8b)는 최종 답변 작성 전담. Ollama의 구조화 출력(format=<JSON Schema>)과
temperature=0으로 변동을 줄이지만 실모델이므로 결정론은 아니다 — 그래서 gate(W6)·
replan(W7)·validator(W8)가 연출이 아니라 실제 방어선으로 동작한다.
fc
비용(cost_usd)은 '가상 단가 × 실측 토큰 수(prompt_eval_count+eval_count)'다 — 로컬
추론이라 실제 과금은 0이지만, W10a의 모델 분리 효과를 같은 잣대로 비교하기 위함이다.

사전 조건: `ollama serve` 실행 중 + `ollama pull qwen3:0.6b qwen3:8b`.

[개선 I-2 — 강의 12장 주의점 "블로킹 호출이 이벤트 루프를 멈춘다"] Base는 route/plan/analyze가 동기
httpx.post를 async 노드 안에서 그대로 불러서, 모델이 답하는 몇 초 동안 서버 전체(다른 사용자 요청, SSE 전송,
/health)가 멈췄다. 이제 모든 호출이 httpx.AsyncClient + await라 기다리는 동안 다른 요청이 진행된다.
[개선 I-11] 동시 호출 수는 agent/limits.py의 llm_slot()으로 제한한다.
[개선 I-4] judge(): 보고서를 쓴 quality 모델과 별개인 fast 모델이 4개 기준으로 검수한다(LLM-as-a-Judge).
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
        "route": {"type": "string", "enum": ["chat", "equipment", "incident"]},
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

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"is_acceptable": {"type": "boolean"}, "reasoning": {"type": "string"}},
    "required": ["is_acceptable", "reasoning"],
}


class OllamaLLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"ollama:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens

    async def _chat(self, system: str, user: str, schema: dict | None = None) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "think": False,   # qwen3 생각 모드 끄기 — 켜 두면 "안녕" 한마디에도 수 분간 생각하다 멈춘 것처럼 보임
            "options": {"temperature": 0.0},
        }
        if schema is not None:
            payload["format"] = schema
        async with llm_slot(), httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        usage = {"model": self.name, "tokens": tokens,
                 "usd": round(tokens / 1000 * self.usd_per_1k, 6)}
        return content, usage

    async def _chat_stream(self, system: str, user: str, on_token=None, model: str | None = None) -> tuple[str, dict]:
        """사용자 대면 텍스트(최종 답변류) 전용 — Ollama NDJSON 스트림을 토큰 단위로
        받아 on_token(조각)을 즉시 호출해 SSE로 실시간 중계할 수 있게 한다. 분류/계획
        등 JSON 스키마 출력(route/plan/analyze)은 부분 출력이 의미가 없어 그대로 _chat을
        쓴다."""
        payload = {
            "model": model or self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "think": False,   # qwen3 생각 모드 끄기 — 켜 두면 "안녕" 한마디에도 수 분간 생각하다 멈춘 것처럼 보임
            "options": {"temperature": 0.0},
        }
        chunks: list[str] = []
        usage = {"model": self.name, "tokens": 0, "usd": 0.0}
        async with llm_slot(), httpx.AsyncClient(timeout=300.0) as client:
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
    async def route(self, query: str) -> tuple[dict, dict]:
        system = ROUTE_SYSTEM   # [After] prompts.py로 이동
        content, usage = await self._chat(system, query, schema=ROUTE_SCHEMA)
        try:
            raw = json.loads(content)
            route = raw.get("route") if raw.get("route") in ("chat", "equipment", "incident") else None
            result = {"route": route or "chat",
                      "needs_analysis": bool(raw.get("needs_analysis", False))}
        except (json.JSONDecodeError, AttributeError):
            # 소형 모델의 형식 붕괴 대비 — 코드 레벨 fallback (W6 정신)
            incident = bool(re.search(INCIDENT_PATTERN, query))
            result = {"route": "incident" if incident else "chat",
                      "needs_analysis": False}
        return result, usage

    @traceable(run_type="llm", name="ollama_fast.plan")
    async def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        system = plan_system(sorted(KNOWN_SYSTEMS))   # [After] prompts.py로 이동
        user = f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"
        content, usage = await self._chat(system, user, schema=PLAN_SCHEMA)

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
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        system = ANALYZE_SYSTEM   # [After] prompts.py로 이동
        facts = analyze_facts(collected)
        return await self._chat(system, facts)

    @traceable(run_type="llm", name="ollama_fast.judge")
    async def judge(self, query: str, collected: list[dict], draft: str) -> tuple[dict, dict]:
        content, usage = await self._chat(JUDGE_SYSTEM, judge_user(query, collected, draft), schema=JUDGE_SCHEMA)
        return parse_judgement(content), usage

    @traceable(run_type="llm", name="ollama_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        system = CHAT_SYSTEM   # [After] prompts.py로 이동
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
        user = (f"## 지금까지의 대화\n{rendered or '(없음)'}\n\n"
                f"## 사용자의 새 메시지\n{query}")
        # 잡담 답변은 8b가 맡는다 — 0.6b는 "오늘 날씨는 흑색입니다"처럼 엉뚱하게 답함(2026-10-22 실측: 8b 1.7~5.3초)
        return await self._chat_stream(system, user, on_token=on_token,
                                       model=os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:8b"))


class OllamaQualityLLM(OllamaLLM):
    """최종 답변 작성 전담 (상대적으로 큰 모델)."""

    @traceable(run_type="llm", name="ollama_quality.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
        system = WRITE_SYSTEM   # [After] prompts.py로 이동
        user = f"## 사용자 요청\n{query}\n\n## 수집된 사실\n" + write_facts(collected) + operator_rule(collected)
        if feedback:
            user += f"\n\n## 이전 초안에 대한 검증 반려 사유 (반드시 반영)\n{feedback}"
        return await self._chat_stream(system, user, on_token=on_token)
