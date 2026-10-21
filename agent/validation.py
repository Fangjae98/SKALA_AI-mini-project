"""[After 추가 — 검수 강화] 보고서 속 숫자가 실제로 수집된 값인지 확인한다.

Before의 검수는 '30자 이상 + 1) 포함'뿐이라, LLM이 수치를 지어내도 합격했다.
After는 보고서에서 단위가 붙은 숫자(kW, kWh, %, °C, V, 분, W/m²)를 모두 뽑아, 질문·계측값·전문 에이전트
점검·매뉴얼 어디에도 없는 숫자가 있으면 반려한다(에너지 보고서의 kW·% 는 틀리면 안 되는 숫자이므로).
"""
import json
import re

NUM_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(kWh|kW|W/m²|%|°C|V|분)")
ANY_NUM = re.compile(r"\d+(?:\.\d+)?")


def _norm(num: str) -> str:
    f = float(num)
    return str(int(f)) if f == int(f) else str(f)


def allowed_numbers(collected: list[dict], query: str) -> set[str]:
    texts = [query]
    for c in collected:
        data = c.get("data")
        texts.append(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
    return {_norm(n) for t in texts for n in ANY_NUM.findall(t)}


def uncited_numbers(draft: str, collected: list[dict], query: str) -> list[str]:
    allowed = allowed_numbers(collected, query)
    bad = []
    for num, unit in NUM_WITH_UNIT.findall(draft):
        if _norm(num) not in allowed and f"{num}{unit}" not in bad:
            bad.append(f"{num}{unit}")
    return bad
