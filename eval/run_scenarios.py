"""[개선 측정] 캡스톤 강의 기준 개선(I-1·I-2·I-3·I-4·I-5·I-11)의 Before/After를 같은 시나리오로 잰다.

eval/run_regression.py가 "답이 맞는가"(매뉴얼 정답률 등)를 잰다면, 이 스크립트는 "장애·동시 사용·재시작 같은
운영 상황에서 버티는가"를 잰다. 코드 폴더를 --root로 받아서, 개선 전 코드(git의 after 태그)와 개선 후 코드에
똑같이 돌릴 수 있게 썼다(개선 전 코드에 없는 기능은 '해당 없음'으로 기록).

실행:
    python3 eval/run_scenarios.py --label 개선후
    git archive after | tar -x -C /tmp/der_before && python3 eval/run_scenarios.py --root /tmp/der_before --label 개선전

시나리오
S1 장애 주입    : EV 충전기 조회만 실패시키고 "EV 충전기 데이터가 안 들어와요" → 답이 나오는가, 실패를 명시하는가
S2 재시작 후 승인 : 승인 대기에서 그래프·저장소를 새로 만든 뒤(=서버 재시작) 승인 → 이어서 끝나는가
S3 품질 저하 요청 : "50자 이내로 짧게" 요청 → 안전 이슈 매뉴얼 2개가 최종 답에 모두 남는가 (심판 끔 vs 켬, 같은 코드)
S4 동시 요청     : 가짜 Ollama 서버(호출당 0.5초)로 질문 2개를 동시에 → 총 시간, 이벤트 루프 최대 멈춤 시간
S5 MCP 동시 조회 : 설비 27건 동시 조회 시간
S6 연관 설비     : "EV 충전기 데이터…" 질문에서 실제로 조사한 설비 목록
"""
import os as _os
# 질문 세트의 정답은 '전체 이상' 시나리오 기준 — 화면 시연용 데이터와 분리
_os.environ.setdefault("DER_MONITORING_FILE", "data/scenarios/full_incident.json")
import argparse
import asyncio
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="측정할 코드 폴더")
parser.add_argument("--label", default="개선후")
args = parser.parse_args()

ROOT = Path(args.root).resolve()
OUT_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.update({"LANGCHAIN_TRACING_V2": "false", "LLM_BACKEND": "fake", "DER_LOG": "off"})
os.environ.pop("DER_FAULT_INJECT", None)

from langgraph.types import Command                 # noqa: E402

from agent.deps import SupervisorDeps               # noqa: E402
from agent.fake_llm import FastLLM, QualityLLM      # noqa: E402
from agent.graph import build_graph                 # noqa: E402
from agent.mcp_client import MCPClient              # noqa: E402
from agent.session_store import SqliteSessionStore  # noqa: E402

HAS_PERSISTENCE = importlib.util.find_spec("agent.persistence") is not None
TMP = Path(tempfile.mkdtemp(prefix="der_scenarios_"))


def _input(q):
    return {"query": q, "thread_id": "s", "history": [], "events": [], "cost_usd": 0.0}


class FailingMCP:
    def __init__(self, inner, fail_targets):
        self._inner, self._fail = inner, set(fail_targets)

    async def call_tool(self, name, arguments, server="ops"):
        if name == "get_equipment_status" and arguments.get("equipment_id") in self._fail:
            raise ConnectionError(f"{arguments['equipment_id']} 계측 서버 응답 없음")
        return await self._inner.call_tool(name, arguments, server)

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def _run(graph, query, decision="approve"):
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    r = await graph.ainvoke(_input(query), cfg)
    while isinstance(r, dict) and r.get("__interrupt__"):
        r = await graph.ainvoke(Command(resume={"decision": decision}), cfg)
    return r


def _deps(llm_fast, llm_quality, mcp):
    return SupervisorDeps(llm_fast, llm_quality, mcp, SqliteSessionStore(TMP / f"{uuid.uuid4().hex}.db"), "v2")


async def s1_fault(mcp):
    graph = build_graph(_deps(FastLLM("f", 0.00015), QualityLLM("q", 0.005), FailingMCP(mcp, {"evc-01"})))
    try:
        r = await _run(graph, "EV 충전기 데이터가 안 들어오는 것 같아요")
    except Exception as e:
        return {"answered": False, "error": f"{type(e).__name__}: {str(e)[:80]}", "failure_stated": False}
    final = r.get("final", "")
    return {"answered": bool(final), "error": None, "failure_stated": "evc-01" in final and "받지 못했" in final,
            "failures": len(r.get("failures", []))}


async def s2_restart(mcp):
    q = "ESS 1호기 온도가 계속 올라가요"
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    deps = _deps(FastLLM("f", 0), QualityLLM("q", 0), mcp)
    try:
        if HAS_PERSISTENCE:
            from agent.persistence import create_checkpointer
            db = TMP / "checkpoints.db"
            async with create_checkpointer("sqlite", db) as cp:
                first = await build_graph(deps, cp).ainvoke(_input(q), cfg)
            async with create_checkpointer("sqlite", db) as cp:   # 재시작 = 저장소·그래프를 새로 만듦
                second = await build_graph(deps, cp).ainvoke(Command(resume={"decision": "approve"}), cfg)
        else:   # 개선 전: build_graph가 메모리 저장소를 스스로 만든다 → 새 그래프 = 새 메모리
            first = await build_graph(deps).ainvoke(_input(q), cfg)
            second = await build_graph(deps).ainvoke(Command(resume={"decision": "approve"}), cfg)
    except Exception as e:
        return {"interrupted": True, "resumed_after_restart": False, "error": f"{type(e).__name__}"}
    ok = isinstance(second, dict) and "[운영자 승인 완료]" in (second.get("final") or "")
    return {"interrupted": "__interrupt__" in first, "resumed_after_restart": ok, "error": None}


async def s3_brevity(mcp):
    from agent.config import RULES
    if "llm_judge" not in RULES["validation"]:
        return {"note": "해당 없음(심판 기능 없음)"}
    q = "ESS 1호기 온도가 계속 올라가요. 답변은 50자 이내로 아주 짧게 해주세요"
    out = {}
    for judge_on in (False, True):
        RULES["validation"]["llm_judge"] = judge_on
        graph = build_graph(_deps(FastLLM("f", 0.00015), QualityLLM("q", 0.005), mcp))
        started = time.perf_counter()
        r = await _run(graph, q)
        final = r.get("final", "")
        out["심판 켬" if judge_on else "심판 끔(코드 검사만)"] = {
            "safety_titles_in_answer": sum(t in final for t in ("ESS 셀 과열", "ESS 충전 상한 초과")),
            "validation_attempts": r.get("validation_attempts"), "cost_usd": r.get("cost_usd"),
            "elapsed_s": round(time.perf_counter() - started, 2)}
    RULES["validation"]["llm_judge"] = True
    return out


class _SlowOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.5)
        if payload.get("stream"):
            body = (json.dumps({"message": {"content": "안녕하세요!"}, "done": True,
                                "prompt_eval_count": 1, "eval_count": 1}) + "\n").encode()
        else:
            body = json.dumps({"message": {"content": '{"route": "chat", "needs_analysis": false}'},
                               "prompt_eval_count": 1, "eval_count": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


async def s4_concurrency(mcp):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import agent.ollama_llm as ol
    ol.OLLAMA_URL = f"http://127.0.0.1:{srv.server_port}/api/chat"
    graph = build_graph(_deps(ol.OllamaFastLLM("stub", 0), ol.OllamaQualityLLM("stub", 0), mcp))
    stall = {"max": 0.0}
    stop = asyncio.Event()

    async def heartbeat():   # 10ms마다 깨어나야 하는데 늦게 깨면 그만큼 이벤트 루프가 막힌 것
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            stall["max"] = max(stall["max"], now - last - 0.01)
            last = now

    hb = asyncio.create_task(heartbeat())
    started = time.perf_counter()
    await asyncio.gather(_run(graph, "안녕하세요"), _run(graph, "반가워요"))
    total = time.perf_counter() - started
    stop.set()
    await hb
    srv.shutdown()
    return {"two_requests_total_s": round(total, 2), "max_event_loop_stall_s": round(stall["max"], 2),
            "note": "LLM 호출 1번 = 0.5초(가짜 Ollama). 질문 1개 = 라우팅 + 답변 = 호출 2번"}


async def s5_mcp_parallel(mcp):
    ids = ["gw-01", "pv-01", "pv-02", "pcc-01", "tr-01", "ess-01", "ess-02", "load-01", "evc-01"] * 3
    started = time.perf_counter()
    results = await asyncio.gather(*(mcp.call_tool("get_equipment_status", {"equipment_id": i}) for i in ids))
    ok = [r["equipment_id"] for r in results] == ids
    return {"calls": len(ids), "elapsed_s": round(time.perf_counter() - started, 3), "results_match": ok}


async def s6_related(mcp):
    graph = build_graph(_deps(FastLLM("f", 0.00015), QualityLLM("q", 0.005), mcp))
    r = await _run(graph, "EV 충전기 데이터가 안 들어오는 것 같아요")
    surveyed = [e["target"] for e in r["events"] if e.get("type") == "TaskCompleted" and e.get("source") == "monitoring"]
    return {"surveyed": surveyed, "gateway_checked": "gw-01" in surveyed}


async def main():
    mcp = MCPClient(flaky_first_tool_call=False)
    await mcp.start()
    results = {}
    try:
        for name, fn in [("S1_장애주입", s1_fault), ("S2_재시작후_승인", s2_restart), ("S3_품질저하요청", s3_brevity),
                         ("S4_동시요청", s4_concurrency), ("S5_MCP_동시조회", s5_mcp_parallel),
                         ("S6_연관설비", s6_related)]:
            try:
                results[name] = await fn(mcp)
            except Exception as e:
                results[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"{name}: {json.dumps(results[name], ensure_ascii=False)}")
    finally:
        await mcp.stop()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"scenarios_{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "root": str(ROOT), "results": results},
                              ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    asyncio.run(main())
