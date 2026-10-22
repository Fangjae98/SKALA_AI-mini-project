"""[개선 I-11 — 강의 12장 주의점] 실모델(Ollama·OpenAI) 동시 호출 상한.

asyncio.gather는 "몇 개까지 한꺼번에 보낼지"를 스스로 제한하지 않는다. 전문 에이전트 4개가 동시에 요약을
요청하고 여러 사용자가 겹치면 로컬 Ollama가 감당 못 할 만큼 요청이 몰린다. 그래서 LLM HTTP 호출은 이 슬롯을
얻은 뒤에만 나간다(상한 = config/rules.json의 concurrency.llm_max_concurrent).

Semaphore는 처음 쓰인 이벤트 루프에 묶이므로, 루프마다 하나씩 따로 만든다(테스트가 asyncio.run을 여러 번
돌려도 "다른 이벤트 루프" 오류가 나지 않게).
"""
import asyncio
import weakref

from .config import RULES

_llm_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def llm_slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _llm_slots.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(RULES["concurrency"]["llm_max_concurrent"])
        _llm_slots[loop] = sem
    return sem
