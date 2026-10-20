"""회귀 테스트 — 같은 질문 10개(eval/testset.json)를 에이전트에 넣고 Before/After 비교용 숫자를 뽑는다.

[Base 대비 추가] Base 학생 배포본에는 평가 스크립트가 없었다.

서버(8100)를 띄우지 않고, server.py와 같은 재료(LLM·MCP·그래프)를 이 스크립트 안에서 직접
만들어 질문을 하나씩 실행한다. LLM은 환경변수 LLM_BACKEND(fake 기본 / ollama / openai)를 따른다.

실행 (프로젝트 폴더에서):
    python3 eval/run_regression.py                 # fake, 결과 이름 'before'
    python3 eval/run_regression.py --label after   # 개선 후 측정
    LLM_BACKEND=ollama python3 eval/run_regression.py --label before

결과: 화면에 표 출력 + eval/results/<label>_<backend>.json 저장
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from agent.deps import SupervisorDeps          # noqa: E402
from agent.graph import build_graph            # noqa: E402
from agent.mcp_client import MCPClient         # noqa: E402
from agent.session_store import SqliteSessionStore  # noqa: E402
from server import _build_llms                 # noqa: E402


def summarize_case(case: dict, result: dict, elapsed: float) -> dict:
    events = result.get("events", [])
    runbooks = [e["target"] for e in events
                if e.get("type") == "TaskCompleted" and e.get("source") == "runbook"]
    targets = [e["target"] for e in events
               if e.get("type") == "TaskCompleted" and e.get("source") == "monitoring"]
    route = result.get("route")
    route_ok = route == case["expected_route"]
    if case["expected_runbook"] is None:
        runbook_ok = not runbooks
    else:
        runbook_ok = case["expected_runbook"] in runbooks
    return {
        "id": case["id"], "query": case["query"],
        "route": route, "route_ok": route_ok,
        "expected_runbook": case["expected_runbook"], "runbooks": runbooks,
        "runbook_ok": runbook_ok,
        "targets": targets,
        "tool_calls": len(targets) + len(runbooks),
        "validation_attempts": result.get("validation_attempts", 0),
        "retry_count": result.get("retry_count", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "elapsed_s": round(elapsed, 2),
    }


async def main(label: str) -> None:
    cases = json.loads((ROOT / "eval" / "testset.json").read_text())
    backend, fast_llm, quality_llm = _build_llms()
    mcp = MCPClient()
    await mcp.start()
    db_path = ROOT / "eval" / "regression_session.db"
    db_path.unlink(missing_ok=True)
    deps = SupervisorDeps(fast_llm=fast_llm, quality_llm=quality_llm, mcp=mcp,
                          session_store=SqliteSessionStore(db_path), version="v2")
    graph = build_graph(deps)

    rows = []
    try:
        for case in cases:
            started = time.perf_counter()
            result = await graph.ainvoke({"query": case["query"], "thread_id": f"reg-{case['id']}",
                                          "history": [], "events": [], "cost_usd": 0.0})
            rows.append(summarize_case(case, result, time.perf_counter() - started))
    finally:
        await mcp.stop()
        deps.session_store.close()
        db_path.unlink(missing_ok=True)

    incident_rows = [r for r in rows if r["expected_runbook"] is not None]
    summary = {
        "label": label, "backend": backend,
        "route_accuracy": f"{sum(r['route_ok'] for r in rows)}/{len(rows)}",
        "runbook_accuracy": f"{sum(r['runbook_ok'] for r in incident_rows)}/{len(incident_rows)}",
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / len(rows), 2),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / len(rows), 2),
        "avg_validation_attempts": round(sum(r["validation_attempts"] for r in rows) / len(rows), 2),
    }

    print(f"\n[{label} / {backend}] 회귀 테스트 결과\n")
    print("| # | 판단 | 매뉴얼(정답) | 매뉴얼(실제) | 결과 | 조사 설비 | 시간(s) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        mark = "O" if (r["route_ok"] and r["runbook_ok"]) else "X"
        print(f"| {r['id']} | {r['route']} | {r['expected_runbook'] or '-'} | "
              f"{', '.join(r['runbooks']) or '-'} | {mark} | {', '.join(r['targets']) or '-'} | "
              f"{r['elapsed_s']} |")
    print("\n요약:", json.dumps(summary, ensure_ascii=False))

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{label}_{backend}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="before", help="결과 이름 (before / after 등)")
    asyncio.run(main(parser.parse_args().label))
