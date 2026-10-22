"""v2 Supervisor — 분산에너지 운영 에이전트의 langgraph.StateGraph.

[Before] 노드 구성·연결은 Base 그대로, 설비 이름 찾기·MCP Tool 이름·매뉴얼 선택 규칙만 분산에너지용으로 교체.
[After]  research = 단계별 전문 에이전트(Multi-Agent) 병렬 호출, replan_check = RAG + 우선순위 매뉴얼 선택,
         validate = 수치 인용 검사 추가, approval = 설비 제어 조치 사람 승인(HITL, interrupt) 노드 추가.
         판단 기준은 모두 config/rules.json에서 읽는다.
[개선 — 캡스톤 강의 기준, doc/설계_결정_기록.md 부록]
  I-1 Error 격리: 설비·전문 에이전트·매뉴얼 하나가 실패해도 나머지로 부분 보고서를 만들고, 실패 부분을 명시한다.
  I-2 모든 LLM 호출을 await(이벤트 루프 블로킹 제거).
  I-4 validate = 코드 검사 + LLM 심판(LLM-as-a-Judge) 혼합 검수.
  I-5 replan_check가 데이터로 확인된 이상의 연관 설비(예: 통신 두절 → 게이트웨이)를 1회 추가 조사.

route → (chat 직행 | incident 파이프라인)
incident: plan(MCP Prompt 절차 반영) → gate(권위 게이트) → research(가드+병렬+재시도)
          → replan_check(부족 시 gap task 후 재조사) → analysis(needs_analysis=True일 때만
          실제 실행) → write → validate(반려 시 재작성) → approval(제어 조치 승인 대기) → finalize

모든 노드는 SupervisorDeps를 functools.partial로 주입받는 순수 함수다(Wave 13).
각 노드는 @traceable로 LangSmith에 계층 트리로 기록된다.
"""
import asyncio
import json
import re
import time
from functools import partial
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from .approval import approval_footer, extract_control_actions
from .config import RULES, incident_types_by_priority
from .deps import SupervisorDeps
from .equipment import DEFAULT_EQUIPMENT, EQUIPMENT, find_mentions
from .diagnosis import judge, problems
from .intent import classify_by_rules, format_detail, format_overview, quick_reply, quick_reply_kind
from .fake_llm import JUDGE_FEEDBACK_TAG
from .retry import ToolError, with_retry
from .rag import data_normal_situation, runbook_excerpt
from .selection import choose_runbooks
from .specialists import build_specialist_graph, specialist_for
from .validation import uncited_numbers


def _extract_system_name(query: str) -> str:
    """MCP Prompt(incident_analysis_prompt)에 넘길 '설비: X' 한 줄용 이름.

    [Base 대비 변경] 영어 서버 이름 정확 일치 → equipment.find_mentions로 설비 ID와
    한국어 별칭(태양광 2호기, ESS, 변압기 …)을 함께 찾는다. 여러 설비가 언급되면 가장
    먼저 나온 하나만 고르고(실제 조사 대상 목록은 plan()의 tasks가 별도로 결정), 하나도
    안 걸리면 단지 전체 전력이 지나가는 연계점(pcc-01)으로 폴백."""
    mentions = find_mentions(query)
    return mentions[0] if mentions else DEFAULT_EQUIPMENT


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    history: list[dict]
    route: str
    needs_analysis: bool
    tasks: list[dict]
    gate_removed: int
    collected: list[dict]
    failures: list[dict]
    replan_rounds: int
    retry_count: int
    parallel_elapsed: float
    draft: str
    validation_attempts: int
    validation_feedback: str
    final: str
    approval: dict
    operator_notes: list[str]   # 승인 단계에서 운영자가 적은 판단(누적) — 재분석의 입력
    last_prompt: str
    cost_usd: float
    events: list[dict]


def _effective_query(state: AgentState) -> str:
    """질문 + 운영자 판단. 운영자가 판단을 적어 돌려보내면 매뉴얼 검색·보고서 작성·검수가 모두 이 문장을 기준으로 다시 돈다."""
    notes = state.get("operator_notes") or []
    if not notes:
        return state["query"]
    return state["query"] + "\n" + "\n".join(f"[운영자 판단] {n}" for n in notes)


def _cost(state: AgentState, usage: dict) -> float:
    return round(state.get("cost_usd", 0.0) + usage["usd"], 6)


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 — 모든 쿼리가 무거운 파이프라인을 탈 필요는 없다.

    [질문 3분류] 잡담(chat) / 설비 조회(equipment) / 매뉴얼 조회(incident).
    인사·도움말·설비 조회처럼 규칙으로 확실한 질문은 AI를 부르지 않고 코드가 바로 분류한다(agent/intent.py)."""
    by_rule = classify_by_rules(state["query"])
    if by_rule:
        return {"route": by_rule, "needs_analysis": False,
                "events": state.get("events", []) + [{"type": "RouteDecided", "route": by_rule, "by": "rule"}]}
    result, usage = await deps.fast_llm.route(state["query"])
    event = {"type": "RouteDecided", "route": result["route"]}
    return {"route": result["route"], "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage),
            "events": state.get("events", []) + [event]}


@traceable(name="chat_answer_node")
async def chat_answer_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 direct 경로 — 히스토리 기반 응답 (Wave 12 세션 증명 지점).

    토큰 단위 스트리밍 — get_stream_writer()는 /chat/stream처럼 custom 모드로 소비될
    때만 실제로 전송되고, /chat(ainvoke)처럼 스트리밍하지 않는 호출에서는 자동으로
    no-op이 된다."""
    writer = get_stream_writer()
    kind = quick_reply_kind(state["query"])
    if kind:   # [질문 3분류] 인사·도움말·매뉴얼 목록 — AI 없이 데이터로 바로 답한다
        reply = quick_reply(kind)
        writer({"type": "AnswerToken", "delta": reply})
        return {"final": reply,
                "events": state["events"] + [{"type": "Finalize", "pipeline": f"chat-{kind}"}]}
    reply, usage = await deps.fast_llm.chat(
        state["query"], state.get("history", []),
        on_token=lambda piece: writer({"type": "AnswerToken", "delta": piece}))
    return {"final": reply, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "Finalize", "pipeline": "chat-direct"}]}


@traceable(name="equipment_node")
async def equipment_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """[질문 3분류] 설비 조회 — 질문에 설비가 있으면 그 설비의 상세 값, 없으면 전체 현황.

    MCP Tool(get_equipment_status)로 지금 값을 읽어 코드가 정리한다(AI가 숫자를 옮기다 틀릴 일이 없다).
    조회에 실패한 설비는 빼고 답하고, 실패했다고 알려 준다(Error 격리)."""
    mentions = find_mentions(state["query"])
    targets = mentions or list(EQUIPMENT.keys())
    extra_events: list[dict] = []

    async def one(eq_id: str):
        try:
            data, _ = await with_retry(deps.mcp.call_tool, "get_equipment_status", {"equipment_id": eq_id})
            return eq_id, data, None
        except ToolError as e:
            return eq_id, None, e.to_dict()

    results = await asyncio.gather(*(one(t) for t in targets))
    ok = [d for _, d, err in results if not err]
    failed = [eq for eq, _, err in results if err]
    for d in ok:   # 분석 과정을 진행 단계로 보여 준다: 기준 대비 판정 → 문제점 분석
        findings = judge(d)
        probs = problems(findings)
        extra_events.append({"type": "StatusJudged", "target": d["equipment_id"], "name": d.get("name"),
                             "checked": len([f for f in findings if f["level"] not in ("info",)]),
                             "abnormal": len(probs), "status": d["status"]})
        if mentions:
            extra_events.append({"type": "ProblemAnalyzed", "target": d["equipment_id"],
                                 "suspected": sorted({f["incident"] for f in probs if f.get("incident")})})
    if mentions:
        parts = [format_detail(d) for d in ok]
        parts += [f"{eq}: 조회 실패 — 데이터를 받지 못했습니다. 현장 확인이 필요합니다." for eq in failed]
        reply = "\n\n".join(parts)   # 다음 단계(대응 질문)는 format_detail이 진단 결과로 만들어 붙인다
    else:
        reply = format_overview(ok, failed)
    events = [{"type": "EquipmentQueried", "targets": targets, "failed": failed}, *extra_events,
              {"type": "Finalize", "pipeline": "equipment", "partial": bool(failed)}]
    return {"final": reply, "events": state["events"] + events}


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 4 — Plan/Task 분해. Wave 2의 MCP Prompt가 표준 절차를 깔아준다.

    resource와 마찬가지로 이름을 어느 서버가 들고 있는지 미리 가정하지 않고,
    find_prompt()로 연결된 서버들을 먼저 조회해 찾은 뒤에만 get_prompt를 보낸다."""
    try:
        prompt_server = await deps.mcp.find_prompt("incident_analysis_prompt")
        procedure = await deps.mcp.get_prompt(
            "incident_analysis_prompt",
            {"system_name": _extract_system_name(state["query"]), "symptom": state["query"]},
            server=prompt_server,
        )
    except Exception as e:   # [개선 I-1] 표준 절차를 못 받아도 계획은 세운다(절차 없이 진행한다고 기록)
        procedure = f"(표준 진단 절차 조회 실패: {type(e).__name__}) 관련 설비 계측값부터 조회하라."
    result, usage = await deps.fast_llm.plan(state["query"], procedure)
    targets = [t["target"] for t in result["tasks"]
               if t["agent"] == "research" and t.get("target")]
    event = {"type": "PlanCreated", "task_count": len(result["tasks"]),
             "targets": targets}
    return {"tasks": result["tasks"], "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="gate_node")
async def gate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 6 — 권위 게이트. needs_analysis=False인데 섞여 들어온 analysis task를
    프롬프트가 아니라 코드로 강제 제거한다."""
    tasks = state["tasks"]
    if not state.get("needs_analysis", False):
        kept = [t for t in tasks if t["agent"] != "analysis"]
        removed = len(tasks) - len(kept)
    else:
        kept, removed = tasks, 0
    event = {"type": "GateApplied", "removed_analysis_tasks": removed}
    return {"tasks": kept, "gate_removed": removed,
            "events": state["events"] + [event]}


@traceable(name="research_node")
async def research_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """[After 변경 — Multi-Agent] Supervisor가 설비 조사를 단계별 전문 에이전트에게 나눠 맡긴다.

    - 계측 조사(monitoring) task: 설비의 단계(stage)를 보고 전문 에이전트별로 묶어, 에이전트들을 동시에 실행
      (각 에이전트 = collect → assess 서브그래프, agent/specialists.py)
    - 매뉴얼(runbook) task: Before와 같이 MCP Resource를 직접 읽는다
    Wave 15 가드(실행할 task가 없으면 건너뜀), Wave 9 재시도, Wave 10b 병렬은 그대로 유지."""
    tasks = state["tasks"]
    pending = [i for i, t in enumerate(tasks) if t["agent"] == "research" and t["status"] == "pending"]
    if not pending:
        return {"events": state["events"] +
                [{"type": "ResearchSkipped", "reason": "pending research task 0개"}]}

    new_tasks = [dict(t) for t in tasks]
    collected = list(state.get("collected", []))
    failures = list(state.get("failures", []))
    events = list(state["events"])
    retry_total = state.get("retry_count", 0)
    cost = state.get("cost_usd", 0.0)
    started = time.perf_counter()

    # 1) 계측 조사 → 전문 에이전트별로 묶기
    groups: dict[str, list[int]] = {}
    for i in pending:
        if tasks[i]["kind"] == "monitoring":
            groups.setdefault(specialist_for(tasks[i]["target"]), []).append(i)
    runbook_idx = [i for i in pending if tasks[i]["kind"] == "runbook"]

    async def run_specialist(key: str, idxs: list[int]) -> tuple[str, list[int], dict | None, str]:
        name = RULES["specialists"][key]["name"]
        try:
            sub = build_specialist_graph(deps)
            out = await sub.ainvoke({"key": key, "name": name,
                                     "targets": [tasks[i]["target"] for i in idxs]})
            return key, idxs, out, ""
        except Exception as e:   # [개선 I-1] 전문 에이전트 하나가 통째로 죽어도 다른 에이전트 결과는 유지
            return key, idxs, None, f"{name} 실행 실패({type(e).__name__}: {e})"

    async def read_runbook(i: int) -> tuple[int, dict | None, dict | None]:
        await asyncio.sleep(0.35)   # Before와 같은 태스크당 고정 지연(지연시간 비교를 공정하게)
        try:
            server, template = await deps.mcp.find_resource_template("get_incident_runbook")
            raw, _ = await with_retry(deps.mcp.read_resource,
                                      template.format(incident_type=tasks[i]["target"]), server=server,
                                      tool_name="get_incident_runbook")
        except ToolError as e:
            return i, None, e.to_dict()
        try:   # [개선 I-8] 매뉴얼 Resource는 {content, last_modified, sha256}을 JSON으로 준다
            doc = json.loads(raw)
        except json.JSONDecodeError:
            doc = {"content": raw, "last_modified": None, "sha256": None}
        return i, doc, None

    spec_results, rb_results = await asyncio.gather(
        asyncio.gather(*(run_specialist(k, v) for k, v in groups.items())),
        asyncio.gather(*(read_runbook(i) for i in runbook_idx)))
    elapsed = round(time.perf_counter() - started, 2)

    for key, idxs, out, crash in spec_results:
        if out is None:
            for i in idxs:
                new_tasks[i]["status"] = "failed"
                fail = {"target": tasks[i]["target"], "code": "AGENT_FAILED", "message": crash, "tool": key}
                failures.append(fail)
                collected.append({"source": "failure", "data": fail})
                events.append({"type": "TaskFailed", "id": tasks[i]["id"], "source": "monitoring",
                               "target": tasks[i]["target"], "code": "AGENT_FAILED", "message": crash})
            continue
        failed_targets = {f["target"] for f in out.get("failures", [])}
        for i in idxs:
            new_tasks[i]["status"] = "failed" if tasks[i]["target"] in failed_targets else "done"
        for f in out.get("failures", []):
            failures.append(f)
            collected.append({"source": "failure", "data": f})
        collected.extend(out["collected"])
        collected.append({"source": "specialist", "data": {
            "agent": out["name"], "targets": out["targets"], "checks": out["checks"],
            "suspected": out["suspected"], "summary": out["summary"]}})
        retry_total += out["retries"]
        cost = round(cost + out.get("cost_usd", 0.0), 6)
        for ev, i in zip(out["task_events"], idxs):
            events.append({**ev, "id": tasks[i]["id"]})
        events.append({"type": "SpecialistReport", "agent": out["name"], "targets": out["targets"],
                       "checks": len(out["checks"]), "suspected": out["suspected"]})
    for i, doc, err in rb_results:
        role = tasks[i].get("role", "primary")
        if err:
            new_tasks[i]["status"] = "failed"
            fail = {"target": f"매뉴얼 {tasks[i]['target']}", **{k: err[k] for k in ("code", "message", "tool")}}
            failures.append(fail)
            collected.append({"source": "failure", "data": fail})
            events.append({"type": "TaskFailed", "id": tasks[i]["id"], "source": "runbook",
                           "target": tasks[i]["target"], "role": role, "code": err["code"],
                           "message": err["message"]})
            continue
        new_tasks[i]["status"] = "done"
        situation, unconfirmed = tasks[i].get("situation"), bool(tasks[i].get("unconfirmed"))
        if unconfirmed and tasks[i].get("situation_by") != "keyword":
            # 운영자가 직접 본 위험 신호(키워드)가 없으면 '데이터 정상일 때' 상황(예: 센서 오류 의심)으로 바꾼다
            situation = data_normal_situation(doc["content"]) or situation
        collected.append({"source": "runbook", "data": runbook_excerpt(doc["content"], situation), "role": role,
                          "situation": situation, "unconfirmed": unconfirmed and tasks[i].get("situation_by") != "keyword",
                          "incident_type": tasks[i]["target"], "last_modified": doc.get("last_modified"),
                          "sha256": doc.get("sha256")})
        events.append({"type": "TaskCompleted", "id": tasks[i]["id"], "source": "runbook",
                       "target": tasks[i]["target"], "role": role, "retries": 0,
                       "last_modified": doc.get("last_modified"), "situation": situation,
                       "unconfirmed": collected[-1].get("unconfirmed", False)})
    return {"tasks": new_tasks, "collected": collected, "failures": failures, "retry_count": retry_total,
            "parallel_elapsed": elapsed, "cost_usd": cost, "events": events}


def _suspected_types(collected: list[dict]) -> list[str]:
    """전문 에이전트들이 데이터로 확인한 이상 유형을 합쳐 우선순위 순서로."""
    found = {t for c in collected if c["source"] == "specialist" for t in c["data"]["suspected"]}
    return [t for t in incident_types_by_priority() if t in found]


@traceable(name="replan_check_node")
async def replan_check_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 7 — 1차 수집 결과 평가. 매뉴얼이 없으면 gap task를 추가해 재조사.

    [After 변경 — RAG + 우선순위] Before는 알람 문장의 키워드를 목록 순서대로 찾아 매뉴얼 1개를 골랐다.
    After는 ① MCP Tool search_runbooks로 '질문 의미'에 가까운 매뉴얼을 찾고 ② 전문 에이전트가 데이터로
    확인한 이상에 가산점을 줘 주 매뉴얼을 정하고 ③ 데이터로 확인된 안전 이슈를 함께 붙인다(agent/selection.py)."""
    has_runbook = any(c["source"] == "runbook" for c in state.get("collected", []))
    rounds = state.get("replan_rounds", 0)
    if has_runbook or rounds >= 1:
        verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": True}
        return {"events": state["events"] + [verdict]}

    suspected = _suspected_types(state.get("collected", []))
    try:
        found, _ = await with_retry(deps.mcp.call_tool, "search_runbooks",
                                    {"query": _effective_query(state), "top_k": RULES["rag"]["top_k"]})
    except ToolError:   # [개선 I-1] 의미 검색이 실패하면 데이터로 확인된 이상의 우선순위로만 고른다
        found = {"hits": [], "backend": "unavailable"}
    choice = choose_runbooks(found["hits"], suspected, found["backend"])
    situation_of = {h["incident_type"]: h.get("situation") for h in found["hits"]}   # 매뉴얼 안에서 찾은 상황 절
    situation_by = {h["incident_type"]: h.get("situation_by") for h in found["hits"]}
    # 데이터 확인: 질문으로 고른 이상이 계측값(전문 에이전트 점검)으로도 확인되는가.
    # 확인 안 되면 설비를 움직이는 조치 대신 '데이터상 정상 — 현장·센서 확인'을 먼저 안내한다.
    data_confirmed = choice["primary"] is None or choice["primary"] in suspected
    gap_tasks = []
    for n, (itype, role) in enumerate([(choice["primary"], "primary")] +
                                      [(t, "safety") for t in choice["extras"]]):
        if itype:
            gap_tasks.append({"id": f"task-gap-runbook-{n}", "agent": "research", "kind": "runbook",
                              "target": itype, "role": role, "situation": situation_of.get(itype),
                              "situation_by": situation_by.get(itype),
                              "unconfirmed": role == "primary" and not data_confirmed,
                              "status": "pending"})
    # [개선 I-5 — 강의 8장 Dynamic Planning] 데이터로 확인된 이상의 연관 설비를 아직 안 봤으면 1회 추가 조사
    surveyed = {t["target"] for t in state["tasks"] if t.get("kind") == "monitoring"}
    extra_targets = []
    for itype in suspected:
        for eq in RULES.get("related_equipment", {}).get(itype, []):
            if eq not in surveyed and eq not in extra_targets:
                extra_targets.append(eq)
    for n, eq in enumerate(extra_targets):
        gap_tasks.append({"id": f"task-gap-related-{n}", "agent": "research", "kind": "monitoring",
                          "target": eq, "status": "pending"})
    verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": False,
               "gap_task": gap_tasks[0]["id"] if gap_tasks else None,
               "target": choice["primary"], "extra": choice["extras"], "method": choice["method"],
               "situation": situation_of.get(choice["primary"]), "data_confirmed": data_confirmed,
               "embedding": found["backend"], "extra_targets": extra_targets,
               "top": [{"type": r["incident_type"], "score": r["score"], "final": r["final"]}
                       for r in choice["ranked"]]}
    return {"tasks": state["tasks"] + gap_tasks, "replan_rounds": rounds + 1,
            "events": state["events"] + [verdict]}


@traceable(name="analysis_node")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 6 보완 — needs_analysis=True가 정당한 경우(모델의 자기모순이 아니라
    실제로 패턴 비교가 필요하다고 판단한 경우)를 위한 실행 경로. gate가 살려둔
    agent=='analysis' task가 있을 때만 수집된 monitoring 결과들을 비교 분석한다."""
    tasks = state["tasks"]
    pending = [t for t in tasks if t["agent"] == "analysis" and t["status"] == "pending"]
    if not pending:
        return {"events": state["events"] +
                [{"type": "AnalysisSkipped", "reason": "analysis task 0개"}]}
    summary, usage = await deps.fast_llm.analyze(state.get("collected", []))
    new_tasks = [dict(t) for t in tasks]
    for t in new_tasks:
        if t["agent"] == "analysis" and t["status"] == "pending":
            t["status"] = "done"
    collected = state.get("collected", []) + [{"source": "analysis", "data": summary}]
    event = {"type": "AnalysisCompleted", "length": len(summary)}
    return {"tasks": new_tasks, "collected": collected,
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="write_node")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8/10a — 고품질 모델로 초안 작성 (1차는 의도적 저품질).

    토큰마다 attempt 번호를 같이 실어 보낸다 — validate가 반려해 재작성(attempt 증가)
    될 때 프론트가 새 말풍선으로 끊어서 보여줄 수 있게 하기 위함(Wave 8 재작성 루프를
    그대로 노출)."""
    attempt = state.get("validation_attempts", 0) + 1
    writer = get_stream_writer()
    draft, usage = await deps.quality_llm.write(
        _effective_query(state), state.get("collected", []),
        attempt, state.get("validation_feedback"),
        on_token=lambda piece: writer(
            {"type": "AnswerToken", "attempt": attempt, "delta": piece}))
    return {"draft": draft, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "DraftWritten", "attempt": attempt,
                                          "length": len(draft)}]}


@traceable(name="validate_node")
async def validate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8 — 코드 기준 검증. 기준값은 config/rules.json의 validation.

    [After 변경 — 검수 강화] Before 기준(30자 이상 + '1)' 포함)에 더해, 보고서 속 숫자(kW·%·°C …)가
    질문·계측값·점검 결과·매뉴얼에 실제로 있는 값인지 확인한다(지어낸 수치 반려)."""
    cfg = RULES["validation"]
    draft = state["draft"]
    attempts = state.get("validation_attempts", 0) + 1
    problems = []
    if len(draft) < cfg["min_length"] or cfg["required_marker"] not in draft:
        problems.append(f"응답이 너무 짧거나 번호 형식이 없습니다({len(draft)}자, 최소 {cfg['min_length']}자). "
                        f"1) 2) 3) 번호로 구체적으로 설명하세요.")
    bad = uncited_numbers(draft, state.get("collected", []), _effective_query(state)) if cfg["check_numbers"] else []
    if bad:
        problems.append(f"수집되지 않은 수치가 있습니다: {', '.join(bad)}. 수집된 사실의 숫자만 그대로 인용하세요.")
    # [개선 I-4 — 강의 9장 LLM-as-a-Judge] 코드 검사를 통과한 초안만 별도 심판 모델에게 보낸다(비용 절약).
    # 형식·수치처럼 규칙으로 확실히 잡을 수 있는 건 코드가, 누락·안전 우선순위 같은 의미 판단은 심판이 맡는다.
    judgement, cost = None, state.get("cost_usd", 0.0)
    if not problems and cfg.get("llm_judge") and hasattr(deps.fast_llm, "judge"):
        try:
            judgement, usage = await deps.fast_llm.judge(_effective_query(state), state.get("collected", []), draft)
            cost = _cost(state, usage)
            if not judgement["is_acceptable"]:
                problems.append(f"{JUDGE_FEEDBACK_TAG} {judgement['reasoning']}")
        except Exception as e:   # 심판이 실패해도 답이 막히지 않게 — 코드 검사 결과만으로 진행
            judgement = {"is_acceptable": True, "reasoning": f"심판 호출 실패({type(e).__name__}) — 코드 검사만 적용"}
    ok = not problems
    verdict = "합격" if ok else "불합격"
    if not ok and attempts >= cfg["max_attempts"]:   # 무한 재작성 방지 — 최대 시도 후 통과 처리
        ok, verdict = True, "합격(최대 시도 도달 — 통과 처리)"
    event = {"type": "Validation", "attempt": attempts, "verdict": verdict, "uncited": bad,
             "judge": judgement}
    return {"validation_attempts": attempts, "validation_feedback": None if ok else " ".join(problems),
            "cost_usd": cost, "events": state["events"] + [event]}


@traceable(name="approval_node")
async def approval_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """[After 추가 — HITL] 보고서에 '설비를 움직이는 조치'가 있으면 그래프를 멈추고 운영자 승인을 기다린다.

    LangGraph interrupt()가 여기서 실행을 멈추고, 화면(또는 API)에서 승인/거부를 보내면
    Command(resume=...)로 이 지점부터 다시 이어진다(멈춘 상태는 checkpointer에 저장)."""
    if not RULES["approval"]["enabled"]:
        return {"events": state["events"] + [{"type": "ApprovalSkipped", "reason": "승인 기능 꺼짐"}]}
    actions = extract_control_actions(state["draft"])
    if not actions:
        return {"approval": {"required": False},
                "events": state["events"] + [{"type": "ApprovalSkipped", "reason": "제어 조치 없음"}]}
    decision = interrupt({"type": "ApprovalRequired", "actions": actions}) or {}
    kind = decision.get("decision", "reject")
    comment = (decision.get("comment") or "").strip()
    if kind == "revise" and comment:
        # [운영자 판단 반영 → 재분석] 보고서만 고쳐 쓰는 게 아니라 문제를 처음부터 다시 분석한다:
        #   ① 최신 계측값 다시 조회(전문 에이전트 재점검) ② 질문+운영자 판단으로 매뉴얼·상황 다시 선택
        #   ③ 판단을 근거로 원인·조치 새로 작성 ④ 다시 검수 ⑤ 다시 승인
        # 조사 대상 설비는 그대로 두고 매뉴얼 task와 이전 수집 결과는 비운다(운영자 판단 기록만 남김).
        notes = (state.get("operator_notes") or []) + [comment]
        tasks = [{**t, "status": "pending"} for t in state["tasks"]
                 if t["agent"] == "research" and t["kind"] == "monitoring"]
        tasks += [t for t in state["tasks"] if t["agent"] != "research"]
        keep = [c for c in state.get("collected", []) if c["source"] == "operator"]
        return {"approval": {"required": True, "revised": True, "comment": comment},
                "operator_notes": notes, "tasks": tasks, "replan_rounds": 0, "failures": [],
                "collected": keep + [{"source": "operator", "data": comment}],
                "validation_feedback": None, "validation_attempts": 0,
                "events": state["events"] + [{"type": "ApprovalRevise", "comment": comment},
                                             {"type": "Reanalysis", "round": len(notes),
                                              "targets": [t["target"] for t in tasks if t.get("kind") == "monitoring"]}]}
    approved = kind == "approve"
    chosen = _chosen_actions(actions, decision.get("actions")) if approved else []
    excluded = [a for a in actions if a not in [c["original"] for c in chosen]] if approved else actions
    return {"approval": {"required": True, "approved": approved, "actions": actions, "comment": comment,
                         "chosen": chosen, "excluded": excluded},
            "events": state["events"] + [{"type": "ApprovalResolved", "approved": approved,
                                          "actions": len(chosen) if approved else len(actions),
                                          "edited": sum(1 for c in chosen if c["text"] != c["original"]),
                                          "excluded": len(excluded) if approved else 0}]}


def _chosen_actions(actions: list[str], picked: list[dict] | None) -> list[dict]:
    """화면에서 고른 조치. picked = [{original, text, checked}] (없으면 전부 원문 그대로 승인).
    목록에 없는 원문은 무시한다(보고서에 없던 조치를 승인 단계에서 끼워 넣지 못하게)."""
    if not picked:
        return [{"original": a, "text": a} for a in actions]
    out = []
    for p in picked:
        if p.get("checked", True) and p.get("original") in actions:
            out.append({"original": p["original"], "text": (p.get("text") or p["original"]).strip()})
    return out


@traceable(name="finalize_node")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    final = state["draft"]
    failures = state.get("failures") or []
    if failures:   # [개선 I-1 — Partial Result] 실패한 부분을 코드가 반드시 덧붙인다(LLM이 빠뜨려도)
        names = ", ".join(f"{f['target']}({f['code']})" for f in failures)
        final += (f"\n\n※ 일부 데이터를 받지 못했습니다: {names}. 해당 부분은 판단에서 제외되어 "
                  "결과의 신뢰도가 낮을 수 있으니 현장 확인이 필요합니다.")
    appr = state.get("approval") or {}
    if appr.get("required"):
        for c in appr.get("chosen", []):          # 운영자가 고친 조치 문구로 보고서를 바꾼다
            if c["text"] != c["original"]:
                head = re.match(r"^\d+\)[^:]*:\s*", c["original"])   # '3) 권장 조치(…): ' 번호 머리말은 유지
                text = c["text"] if not head or c["text"].startswith(head.group(0)) else head.group(0) + c["text"]
                final = final.replace(c["original"], f"{text} (운영자 수정)")
        if appr.get("approved"):
            for a in appr.get("excluded", []):    # 체크를 뺀 조치는 시행하지 않는다고 표시
                final = final.replace(a, f"{a} (운영자 제외 — 시행하지 않음)")
        final += "\n\n" + approval_footer(appr["actions"], appr["approved"], appr.get("comment", ""),
                                          chosen=appr.get("chosen"), excluded=appr.get("excluded"))
    return {"final": final,
            "events": state["events"] + [{"type": "Finalize", "pipeline": "incident-v2",
                                          "partial": bool(failures)}]}


def _after_approval(state: AgentState) -> str:
    return "research" if (state.get("approval") or {}).get("revised") else "finalize"


def _after_route(state: AgentState) -> str:
    return {"chat": "chat_answer", "equipment": "equipment"}.get(state["route"], "plan")


def _after_replan_check(state: AgentState) -> str:
    last = state["events"][-1]
    return "analysis" if last.get("is_sufficient") else "research"


def _after_validate(state: AgentState) -> str:
    return "approval" if state.get("validation_feedback") is None else "write"


def build_graph(deps: SupervisorDeps, checkpointer=None):
    """[After 변경] 승인 노드(approval) 추가. interrupt()로 멈춘 상태를 저장하려면 checkpointer가 필요해
    기본으로 메모리 저장소(InMemorySaver)를 붙인다(서버를 재시작하면 대기 중인 승인은 사라짐 — 한계)."""
    g = StateGraph(AgentState)
    node = lambda fn: partial(fn, deps=deps)   # Wave 13 — 명시적 의존성 주입
    g.add_node("route", node(route_node))
    g.add_node("chat_answer", node(chat_answer_node))
    g.add_node("equipment", node(equipment_node))
    g.add_node("plan", node(plan_node))
    g.add_node("gate", node(gate_node))
    g.add_node("research", node(research_node))
    g.add_node("replan_check", node(replan_check_node))
    g.add_node("analysis", node(analysis_node))
    g.add_node("write", node(write_node))
    g.add_node("validate", node(validate_node))
    g.add_node("approval", node(approval_node))
    g.add_node("finalize", node(finalize_node))

    g.set_entry_point("route")
    g.add_conditional_edges("route", _after_route,
                            {"chat_answer": "chat_answer", "equipment": "equipment", "plan": "plan"})
    g.add_edge("chat_answer", END)
    g.add_edge("equipment", END)
    g.add_edge("plan", "gate")
    g.add_edge("gate", "research")
    g.add_edge("research", "replan_check")
    g.add_conditional_edges("replan_check", _after_replan_check,
                            {"analysis": "analysis", "research": "research"})
    g.add_edge("analysis", "write")
    g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"approval": "approval", "write": "write"})
    g.add_conditional_edges("approval", _after_approval, {"research": "research", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
