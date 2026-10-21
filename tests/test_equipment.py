"""설비 찾기(한국어 별칭)와 계측값 표시 형식 테스트."""
from agent.equipment import find_mentions, format_metrics


def test_개별_별칭으로_설비를_찾는다():
    assert find_mentions("태양광 2호기 발전량이 낮아요") == ["pv-02"]


def test_묶음_별칭은_묶음_전체를_찾는다():
    assert find_mentions("ESS 상태 알려줘") == ["ess-01", "ess-02"]


def test_개별_설비가_있으면_묶음_별칭은_무시한다():
    assert find_mentions("ESS 1호기 온도") == ["ess-01"]


def test_여러_설비는_질문에_나온_순서대로():
    assert find_mentions("변압기랑 ESS 1호기 같이 봐줘") == ["tr-01", "ess-01"]


def test_설비가_없으면_빈_목록():
    assert find_mentions("오늘 날씨 어때?") == []


def test_계측값을_한국어와_단위로_표시한다():
    assert format_metrics({"output_kw": 96, "pr_percent": 46}) == "출력 96kW, 발전 효율(PR) 46%"
