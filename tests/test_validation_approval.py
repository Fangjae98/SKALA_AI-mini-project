"""검수(지어낸 수치 찾기)와 사람 승인 대상 문장 찾기 테스트."""
from agent.approval import extract_control_actions
from agent.validation import uncited_numbers

COLLECTED = [{"source": "monitoring", "data": {"metrics": {"soc_percent": 86, "charge_kw": 70}}}]


def test_수집된_수치만_쓰면_통과():
    assert uncited_numbers("1) SOC 86%, 충전 70kW", COLLECTED, "질문") == []


def test_지어낸_수치는_잡아낸다():
    assert uncited_numbers("1) SOC 91%, 충전 70kW", COLLECTED, "질문") == ["91%"]


def test_질문에_나온_수치는_허용():
    assert uncited_numbers("80%를 넘었습니다", COLLECTED, "80%를 넘었는데 괜찮나요?") == []


def test_승인표시나_제어문구가_있는_문장만_찾는다():
    draft = ("1) 현재 상태: SOC 86%. 2) 원인: 충전 상한 초과. "
             "3) 즉시 충전을 중지한다(운영자 승인 필요). 4) 결과를 기록한다.")
    actions = extract_control_actions(draft)
    assert len(actions) == 1 and "충전을 중지" in actions[0]


def test_제어_조치가_없으면_빈_목록():
    assert extract_control_actions("1) 상태 정상. 2) 기록만 남긴다.") == []
