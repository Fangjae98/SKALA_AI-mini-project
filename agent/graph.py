"""v2 Supervisor — 분산에너지 운영 에이전트의 langgraph.StateGraph.

[Before] 노드 구성·연결은 Base 그대로, 설비 이름 찾기·MCP Tool 이름·매뉴얼 선택 규칙만 분산에너지용으로 교체.
[After]  research = 단계별 전문 에이전트(Multi-Agent) 병렬 호출, replan_check = RAG + 우선순위 매뉴얼 선택,
         validate = 수치 인용 검사 추가, approval = 설비 제어 조치 사람 승인(HITL, interrupt) 노드 추가.
         판단 기준은 모두 config/rules.json에서 읽는다.

route → (chat 직행 | incident 파이프라인)
incident: plan(MCP Prompt 절차 반영) → gate(권위 게이트) → research(가드+병렬+재시도)
          → replan_check(부족 시 gap task 후 재조사) → analysis(needs_analysis=True일 때만
          실제 실행) → write → validate(반려 시 재작성) → approval(제어 조치 승인 대기) → finalize

모든 노드는 SupervisorDeps를 functools.partial로 주입받는 순수 함수다(Wave 13).
각 노드는 @traceable로 LangSmith에 계층 트리로 기록된다.
"""
import asyncio
import time
from functools import partial
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from .approval import approval_footer, extract_control_actions
from .config import RULES, incident_types_by_priority
from .deps import SupervisorDeps
from .equipment import DEFAULT_EQUIPMENT, find_mentions
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
    replan_rounds: int
    retry_count: int
    parallel_elapsed: float
    draft: str
    validation_attempts: int
    validation_feedback: str
    final: str
    approval: dict
    last_prompt: str
    cost_usd: float
    events: list[dict]


def _cost(state: AgentState, usage: dict) -> float:
    return round(state.get("cost_usd", 0.0) + usage["usd"], 6)


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 — 모든 쿼리가 무거운 파이프라인을 탈 필요는 없다."""
    result, usage = deps.fast_llm.route(state["query"])
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
    reply, usage = await deps.fast_llm.chat(
        state["query"], state.get("history", []),
        on_token=lambda piece: writer({"type": "AnswerToken", "delta": piece}))
    return {"final": reply, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "Finalize", "pipeline": "chat-direct"}]}


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 4 — Plan/Task 분해. Wave 2의 MCP Prompt가 표준 절차를 깔아준다.

    resource와 마찬가지로 이름을 어느 서버가 들고 있는지 미리 가정하지 않고,
    find_prompt()로 연결된 서버들을 먼저 조회해 찾은 뒤에만 get_prompt를 보낸다."""
    prompt_server = await deps.mcp.find_prompt("incident_analysis_prompt")
    procedure = await deps.mcp.get_prompt(
        "incident_analysis_prompt",
        {"system_name": _extract_system_name(state["query"]), "symptom": state["query"]},
        server=prompt_server,
    )
    result, usage = deps.fast_llm.plan(state["query"], procedure)
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

    async def run_specialist(key: str, idxs: list[int]) -> tuple[str, list[int], dict]:
        sub = build_specialist_graph(deps)
        out = await sub.ainvoke({"key": key, "name": RULES["specialists"][key]["name"],
                                 "targets": [tasks[i]["target"] for i in idxs]})
        return key, idxs, out

    async def read_runbook(i: int) -> tuple[int, str]:
        await asyncio.sleep(0.35)   # Before와 같은 태스크당 고정 지연(지연시간 비교를 공정하게)
        server, template = await deps.mcp.find_resource_template("get_incident_runbook")
        return i, await deps.mcp.read_resource(template.format(incident_type=tasks[i]["target"]),
                                               server=server)

    spec_results, rb_results = await asyncio.gather(
        asyncio.gather(*(run_specialist(k, v) for k, v in groups.items())),
        asyncio.gather(*(read_runbook(i) for i in runbook_idx)))
    elapsed = round(time.perf_counter() - started, 2)

    for key, idxs, out in spec_results:
        for i in idxs:
            new_tasks[i]["status"] = "done"
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
    for i, text in rb_results:
        new_tasks[i]["status"] = "done"
        collected.append({"source": "runbook", "data": text, "role": tasks[i].get("role", "primary"),
                          "incident_type": tasks[i]["target"]})
        events.append({"type": "TaskCompleted", "id": tasks[i]["id"], "source": "runbook",
                       "target": tasks[i]["target"], "role": tasks[i].get("role", "primary"),
                       "retries": 0})
    return {"tasks": new_tasks, "collected": collected, "retry_count": retry_total,
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
    found = await deps.mcp.call_tool("search_runbooks",
                                     {"query": state["query"], "top_k": RULES["rag"]["top_k"]})
    choice = choose_runbooks(found["hits"], suspected, found["backend"])
    gap_tasks = []
    for n, (itype, role) in enumerate([(choice["primary"], "primary")] +
                                      [(t, "safety") for t in choice["extras"]]):
        if itype:
            gap_tasks.append({"id": f"task-gap-runbook-{n}", "agent": "research", "kind": "runbook",
                              "target": itype, "role": role, "status": "pending"})
    verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": False,
               "gap_task": gap_tasks[0]["id"] if gap_tasks else None,
               "target": choice["primary"], "extra": choice["extras"], "method": choice["method"],
               "embedding": found["backend"],
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
    summary, usage = deps.fast_llm.analyze(state.get("collected", []))
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
        state["query"], state.get("collected", []),
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
    bad = uncited_numbers(draft, state.get("collected", []), state["query"]) if cfg["check_numbers"] else []
    if bad:
        problems.append(f"수집되지 않은 수치가 있습니다: {', '.join(bad)}. 수집된 사실의 숫자만 그대로 인용하세요.")
    ok = not problems
    verdict = "합격" if ok else "불합격"
    if not ok and attempts >= cfg["max_attempts"]:   # 무한 재작성 방지 — 최대 시도 후 통과 처리
        ok, verdict = True, "합격(최대 시도 도달 — 통과 처리)"
    event = {"type": "Validation", "attempt": attempts, "verdict": verdict, "uncited": bad}
    return {"validation_attempts": attempts, "validation_feedback": None if ok else " ".join(problems),
            "events": state["events"] + [event]}


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
    decision = interrupt({"type": "ApprovalRequired", "actions": actions})
    approved = (decision or {}).get("decision") == "approve"
    comment = (decision or {}).get("comment", "")
    return {"approval": {"required": True, "approved": approved, "actions": actions, "comment": comment},
            "events": state["events"] + [{"type": "ApprovalResolved", "approved": approved,
                                          "actions": len(actions)}]}


@traceable(name="finalize_node")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    final = state["draft"]
    appr = state.get("approval") or {}
    if appr.get("required"):
        final += "\n\n" + approval_footer(appr["actions"], appr["approved"], appr.get("comment", ""))
    return {"final": final,
            "events": state["events"] + [{"type": "Finalize", "pipeline": "incident-v2"}]}


def _after_route(state: AgentState) -> str:
    return "chat_answer" if state["route"] == "chat" else "plan"


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
                            {"chat_answer": "chat_answer", "plan": "plan"})
    g.add_edge("chat_answer", END)
    g.add_edge("plan", "gate")
    g.add_edge("gate", "research")
    g.add_edge("research", "replan_check")
    g.add_conditional_edges("replan_check", _after_replan_check,
                            {"analysis": "analysis", "research": "research"})
    g.add_edge("analysis", "write")
    g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"approval": "approval", "write": "write"})
    g.add_edge("approval", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
