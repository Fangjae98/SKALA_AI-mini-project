"""RAG 매뉴얼 검색(TF-IDF)과 매뉴얼 선택 규칙 테스트."""
import pytest

from agent.rag import RunbookIndex, TfidfEmbedder
from agent.selection import choose_runbooks


@pytest.fixture(scope="module")
def index():
    return RunbookIndex(TfidfEmbedder())


@pytest.mark.parametrize("query,expected", [
    ("ESS 1호기 온도가 계속 올라가요", "ess-overheat"),
    ("DR 감축 요청이 왔어요, 어떻게 대응하죠?", "dr-event"),
    ("출력제어 지시가 들어왔는데 어떻게 대응해야 하나요?", "curtailment"),
    ("변압기 쪽 전압이 높다는 경보가 떴어요", "voltage-rise"),
])
def test_질문_의미로_매뉴얼을_찾는다(index, query, expected):
    assert index.search(query, 1)[0]["incident_type"] == expected


def test_데이터_근거_가산점으로_단어_착시를_바로잡는다(index):
    hits = index.search("EV 충전기 데이터가 안 들어오는 것 같아요", 9)
    assert hits[0]["incident_type"] == "ess-soc-limit"            # 의미 점수만 보면 '충전' 때문에 착시
    choice = choose_runbooks(hits, ["comm-loss"], "tfidf")       # 저장·공급 에이전트가 데이터로 확인
    assert choice["primary"] == "comm-loss"


def test_질문과_다른_안전이슈는_함께_안내한다(index):
    hits = index.search("ESS 1호기 충전율이 80%를 넘었는데 괜찮나요?", 9)
    choice = choose_runbooks(hits, ["ess-overheat", "ess-soc-limit"], "tfidf")
    assert choice["primary"] == "ess-soc-limit"
    assert choice["extras"] == ["ess-overheat"]


def test_의도가_불분명하면_우선순위로_고른다():
    hits = [{"incident_type": "dr-event", "title": "", "stage": "", "score": 0.0}]
    choice = choose_runbooks(hits, ["comm-loss", "peak-demand"], "tfidf")
    assert choice["method"] == "priority" and choice["primary"] == "comm-loss"
