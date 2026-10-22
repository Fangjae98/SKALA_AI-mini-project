"""전문 에이전트의 계측값 점검 규칙 테스트 — 숫자 판단은 LLM이 아니라 코드가 한다."""
import json

from agent.config import ROOT
from agent.specialists import specialist_for, stage_checks

DATA = json.loads((ROOT / "data" / "scenarios" / "full_incident.json").read_text())


def status(eq_id):
    return {"equipment_id": eq_id, **DATA[eq_id]}


def types(eq_id):
    return {t for _, t in stage_checks(status(eq_id)) if t}


def test_설비를_단계별_전문_에이전트에_배정한다():
    assert specialist_for("pv-01") == "generation"
    assert specialist_for("pcc-01") == "grid"
    assert specialist_for("ess-02") == "storage"
    assert specialist_for("gw-01") == "data"


def test_ESS1은_충전상한과_과열을_둘_다_찾는다():
    assert types("ess-01") == {"ess-soc-limit", "ess-overheat"}


def test_ESS2는_방전하한을_찾는다():
    assert types("ess-02") == {"ess-soc-low"}


def test_태양광2호기_발전효율_저하():
    assert types("pv-02") == {"pv-underperform"}


def test_변압기_전압상승_계산():
    checks = stage_checks(status("tr-01"))
    assert checks == [("tr-01 모선 전압 403V = 정격 380V 대비 +6.1%", "voltage-rise")]


def test_연계점_피크_비율_계산():
    assert ("pcc-01 오늘 최대수요 472kW = 계약전력 500kW의 94.4%", "peak-demand") in stage_checks(status("pcc-01"))


def test_오래된_데이터는_통신두절로_본다():
    assert types("evc-01") == {"comm-loss"}


def test_정상_설비는_점검_결과가_없다():
    assert stage_checks(status("load-01")) == []
