"""설비별 상태 판단 기준 — 계측값만 보고 코드가 정상/주의/이상을 판정하고, 문제점과 대응 질문을 만든다.

화면 시연용 데이터(data/monitoring.json): 태양광 2호기·ESS 2호기·EV 충전기·계측 게이트웨이 4대만 이상."""
import json

from agent.config import ROOT
from agent.diagnosis import generated_alerts, judge, overall_status, suggested_question

DEMO = json.loads((ROOT / "data" / "monitoring.json").read_text())


def status(eq_id, **metrics):
    data = json.loads(json.dumps(DEMO[eq_id]))
    data["metrics"].update(metrics)
    return {"equipment_id": eq_id, **data}


def test_시연_데이터는_4대만_이상():
    bad = {eq for eq in DEMO if overall_status(judge(status(eq))) != "healthy"}
    assert bad == {"pv-02", "ess-02", "evc-01", "gw-01"}


def test_데이터_파일에_알람을_적지_않아도_기준으로_만든다():
    alerts = generated_alerts(judge(status("pv-02")))
    assert alerts == ["발전 효율(PR) 46% (경보 기준 60% 미만)", "스트링 1개 전류 0A (전체 4개 중)"]


def test_주의와_경보를_구분한다():
    assert overall_status(judge(status("ess-01", cell_temp_max_c=42))) == "warning"     # 40 이상 주의
    assert overall_status(judge(status("ess-01", cell_temp_max_c=46))) == "degraded"    # 45 이상 경보


def test_일사량이_낮으면_발전_효율은_판정_보류():
    f = next(x for x in judge(status("pv-02", irradiance_wm2=120)) if x["metric"] == "pr_percent")
    assert f["level"] == "hold" and "날씨" in f["meaning"]


def test_대응_질문은_알람_복사가_아니라_진단_결과로_만든다():
    q = suggested_question(status("gw-01"), judge(status("gw-01")))
    assert q.startswith("계측 게이트웨이(gw-01)에서 연결 설비 8/9대") and q.endswith("원인과 대응 방법을 알려주세요.")


def test_정상_설비는_대응_질문이_없다():
    assert suggested_question(status("load-01"), judge(status("load-01"))) is None


# ── 운영자 말과 데이터가 다를 때 (시연 데이터: ESS 1호기는 정상 31°C) ─────────────────────────────
def _ask_demo(query, tmp_path, monkeypatch):
    import asyncio
    import uuid
    from agent.deps import SupervisorDeps
    from agent.fake_llm import FastLLM, QualityLLM
    from agent.graph import build_graph
    from agent.mcp_client import MCPClient
    from agent.session_store import SqliteSessionStore
    monkeypatch.setenv("DER_MONITORING_FILE", "data/monitoring.json")   # MCP 서버(별도 프로세스)가 시연 데이터를 읽게

    async def run():
        mcp = MCPClient(flaky_first_tool_call=False)
        await mcp.start()
        try:
            g = build_graph(SupervisorDeps(FastLLM("f", 0), QualityLLM("q", 0), mcp,
                                           SqliteSessionStore(tmp_path / "s.db"), "v2"))
            return await g.ainvoke({"query": query, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0},
                                   {"configurable": {"thread_id": uuid.uuid4().hex}})
        finally:
            await mcp.stop()
    return asyncio.run(run())


def test_데이터로_확인_안되면_설비를_멈추지_않고_확인부터(tmp_path, monkeypatch):
    r = _ask_demo("ESS 1호기 온도가 계속 올라가요", tmp_path, monkeypatch)
    assert not r.get("__interrupt__")                                   # 승인 요청(설비 정지) 없음
    assert "데이터상 이상이 확인되지 않음" in r["final"] and "셀 최고온도 31°C → 정상" in r["final"]
    assert "열화상" in r["final"]                                        # '데이터 정상' 상황(센서 확인) 절차


def test_운영자가_직접_본_위험_신호는_데이터가_정상이어도_대응(tmp_path, monkeypatch):
    r = _ask_demo("ESS 1호기 쪽 ESS실에서 타는 냄새가 나요", tmp_path, monkeypatch)
    assert r.get("__interrupt__") and "대피" in r["draft"]               # 화재 징후 → 비상 대응 + 승인
