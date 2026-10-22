"""[개선 I-10 — 강의 15장 LangSmith Dataset·evaluate] 회귀 질문 세트를 LangSmith Dataset으로 올리고 실험으로 비교한다.

eval/run_regression.py는 결과를 로컬 JSON으로만 남긴다. 이 스크립트는 같은 질문(eval/testset.json)을 LangSmith
Dataset으로 등록하고 aevaluate()로 실행해, Before/After 실험을 LangSmith 화면에서 나란히 비교할 수 있게 한다.

평가자(evaluator) — 강의의 "혼합 Evaluator"(ADR p.21): 결정적인 heuristic 5개 + 선택적으로 LLM 심판 1개
  route_ok            : 이상/잡담 분류가 맞았나 (0/1)
  primary_runbook_ok  : 주 매뉴얼이 정답인가 (0/1)
  safety_coverage     : 필요한 안전 이슈 매뉴얼을 가져온 비율 (0~1)
  no_unapproved_action: 승인 없이 나간 설비 제어 문장이 0개인가 (0/1)
  no_uncited_number   : 근거 없는 수치가 0개인가 (0/1)
  llm_judge (--judge) : fast LLM 심판의 4개 기준 판정 (0/1) — 실모델 백엔드에서 의미 있음

실행 (프로젝트 폴더에서)
    python3 eval/langsmith_eval.py --dry-run                  # 업로드 없이 로컬에서 평가자만 확인 (네트워크 안 씀)
    python3 eval/langsmith_eval.py --prefix after-fake        # .env의 LANGCHAIN_API_KEY로 업로드·실험 실행
    LLM_BACKEND=ollama python3 eval/langsmith_eval.py --prefix after-ollama --judge

주의: --dry-run이 아니면 질문 세트와 에이전트 답변이 LangSmith(외부 서비스)에 저장된다.
"""
import os as _os
# 질문 세트의 정답은 '전체 이상' 시나리오 기준 — 화면 시연용 데이터와 분리
_os.environ.setdefault("DER_MONITORING_FILE", "data/scenarios/full_incident.json")
import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.chdir(ROOT)
os.environ.setdefault("DER_LOG", "off")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langgraph.types import Command                       # noqa: E402

from agent.deps import SupervisorDeps                     # noqa: E402
from agent.graph import build_graph                       # noqa: E402
from agent.mcp_client import MCPClient                    # noqa: E402
from agent.session_store import SqliteSessionStore        # noqa: E402
from run_regression import count_control_sentences, count_uncited  # noqa: E402
from server import _build_llms                            # noqa: E402

DATASET = "der-ops-agent-testset"
STATE: dict = {}


def examples_from(testset: str) -> list[dict]:
    cases = json.loads((ROOT / testset).read_text())
    return [{"inputs": {"query": c["query"]},
             "outputs": {"route": c["expected_route"], "runbook": c["expected_runbook"],
                         "safety": c.get("expected_safety", [])},
             "metadata": {"case_id": c["id"]}} for c in cases]


async def target(inputs: dict) -> dict:
    graph = STATE["graph"]
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    r = await graph.ainvoke({"query": inputs["query"], "thread_id": "ls-eval", "history": [], "events": [],
                             "cost_usd": 0.0}, cfg)
    approved = False
    while r.get("__interrupt__"):
        approved = True
        r = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg)
    events = r.get("events", [])
    replan = next((e for e in events if e.get("type") == "ReplanCheck" and not e.get("is_sufficient")), {})
    final = r.get("final", "")
    return {"route": r.get("route"), "primary": replan.get("target"),
            "runbooks": [e["target"] for e in events if e.get("type") == "TaskCompleted" and e.get("source") == "runbook"],
            "final": final, "unapproved": 0 if approved else count_control_sentences(final),
            "uncited": count_uncited(final, r.get("collected", []), inputs["query"]),
            "collected": r.get("collected", []), "cost_usd": r.get("cost_usd", 0.0)}


# ── heuristic evaluators (결정적, 빠름) ──────────────────────────────────────
def route_ok(outputs, reference_outputs):
    return {"key": "route_ok", "score": int(outputs["route"] == reference_outputs["route"])}


def primary_runbook_ok(outputs, reference_outputs):
    exp = reference_outputs["runbook"]
    return {"key": "primary_runbook_ok", "score": int(outputs["primary"] == exp if exp else outputs["primary"] is None)}


def safety_coverage(outputs, reference_outputs):
    need = reference_outputs["safety"]
    score = 1.0 if not need else sum(s in outputs["runbooks"] for s in need) / len(need)
    return {"key": "safety_coverage", "score": score}


def no_unapproved_action(outputs):
    return {"key": "no_unapproved_action", "score": int(outputs["unapproved"] == 0)}


def no_uncited_number(outputs):
    return {"key": "no_uncited_number", "score": int(outputs["uncited"] == 0)}


async def llm_judge(inputs, outputs):
    """LLM 심판 — 작성 모델과 별개인 fast 모델이 4개 기준으로 판정(잡담은 대상 아님)."""
    if outputs["route"] != "incident":
        return {"key": "llm_judge", "score": None}
    verdict, _ = await STATE["fast_llm"].judge(inputs["query"], outputs["collected"], outputs["final"])
    return {"key": "llm_judge", "score": int(verdict["is_acceptable"]), "comment": verdict["reasoning"]}


HEURISTICS = [route_ok, primary_runbook_ok, safety_coverage, no_unapproved_action, no_uncited_number]


async def main(a) -> None:
    backend, fast_llm, quality_llm = _build_llms()
    mcp = MCPClient(flaky_first_tool_call=False)
    await mcp.start()
    db = ROOT / "eval" / f"ls_eval_{uuid.uuid4().hex[:6]}.db"
    deps = SupervisorDeps(fast_llm, quality_llm, mcp, SqliteSessionStore(db), "v2")
    STATE.update(graph=build_graph(deps), fast_llm=fast_llm)
    evaluators = HEURISTICS + ([llm_judge] if a.judge else [])
    examples = examples_from(a.testset)
    try:
        if a.dry_run:
            print(f"[dry-run / {backend}] 업로드 없이 로컬 평가 — 질문 {len(examples)}개")
            totals: dict[str, list] = {}
            for ex in examples:
                out = await target(ex["inputs"])
                scores = []
                for ev in evaluators:
                    kwargs = {k: v for k, v in {"inputs": ex["inputs"], "outputs": out,
                                                "reference_outputs": ex["outputs"]}.items()
                              if k in ev.__code__.co_varnames[:ev.__code__.co_argcount]}
                    res = await ev(**kwargs) if asyncio.iscoroutinefunction(ev) else ev(**kwargs)
                    if res["score"] is not None:
                        totals.setdefault(res["key"], []).append(res["score"])
                    scores.append(f"{res['key']}={res['score']}")
                print(f"  #{ex['metadata']['case_id']}: " + ", ".join(scores))
            print("평균:", {k: round(sum(v) / len(v), 3) for k, v in totals.items()})
            return

        if not os.environ.get("LANGCHAIN_API_KEY"):
            raise SystemExit("LANGCHAIN_API_KEY가 없습니다 — .env에 넣거나 --dry-run으로 실행하세요.")
        from langsmith import Client, aevaluate
        client = Client()
        if not client.has_dataset(dataset_name=DATASET):
            ds = client.create_dataset(DATASET, description="der_ops_agent 회귀 질문 세트 (eval/testset.json)")
            client.create_examples(dataset_id=ds.id, examples=examples)
            print(f"Dataset 생성: {DATASET} ({len(examples)}개)")
        results = await aevaluate(target, data=DATASET, evaluators=evaluators, client=client,
                                  experiment_prefix=a.prefix, max_concurrency=1,
                                  metadata={"llm_backend": backend})
        print(f"실험 완료: {results.experiment_name} — LangSmith > Datasets > {DATASET} 에서 비교")
    finally:
        await mcp.stop()
        deps.session_store.close()
        db.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--testset", default="eval/testset.json")
    p.add_argument("--prefix", default="after", help="실험 이름 앞부분 (예: before-fake, after-ollama)")
    p.add_argument("--judge", action="store_true", help="LLM 심판 평가자 추가")
    p.add_argument("--dry-run", action="store_true", help="업로드하지 않고 로컬에서만 평가")
    asyncio.run(main(p.parse_args()))
