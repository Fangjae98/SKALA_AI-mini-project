# 회귀 테스트 질문 목록 — Before/After 비교용

`eval/testset.json`과 같은 내용이다. `python3 eval/run_regression.py --label <이름>`으로 실행한다.

| # | 질문 | 정답 판단 | 정답 매뉴얼 | 조사해야 할 설비 | 테스트 목적 |
|---|---|---|---|---|---|
| 1 | 태양광 2호기 발전량이 평소보다 너무 낮아요, 원인 확인해주세요 | incident | pv-underperform | pv-02 | 한국어 별칭(태양광 2호기 → pv-02) 인식, 생산 단계 |
| 2 | 출력제어 지시가 들어왔는데 어떻게 대응해야 하나요? | incident | curtailment | pv-01, pv-02 | 설비 이름 없이 묶음 별칭(출력제어 → 태양광 전체) 인식 |
| 3 | 변압기 쪽 전압이 높다는 경보가 떴어요 | incident | voltage-rise | tr-01 | 이동 단계 |
| 4 | 오늘 오전 피크가 계약전력에 거의 닿았어요. 원인 분석해줘 | incident | peak-demand | pcc-01 | 이동 단계, 연계점 |
| 5 | ESS 1호기 온도가 계속 올라가요 | incident | ess-overheat | ess-01 | **알람 2개(충전 상한·과열) 중 안전 이슈를 고르는지** |
| 6 | ESS 1호기 충전율이 80%를 넘었는데 괜찮나요? | incident | ess-soc-limit | ess-01 | 저장 단계, 같은 설비에서 질문 의도에 맞는 매뉴얼 선택 |
| 7 | ESS 2호기가 거의 방전됐어요 | incident | ess-soc-low | ess-02 | 공급 단계 |
| 8 | DR 감축 요청이 왔어요, 어떻게 대응하죠? | incident | dr-event | pcc-01 | **연계점 알람 3개(피크·역송·DR) 중 질문에 맞는 것을 고르는지** |
| 9 | EV 충전기 데이터가 안 들어오는 것 같아요 | incident | comm-loss | evc-01 | 수집(데이터) 층 |
| 10 | 안녕, 너는 뭘 할 수 있어? | chat | (없음) | (없음) | 잡담은 무거운 파이프라인을 타지 않는지 |

## Before 결과 (fake 백엔드, 2026-10-20)

- 판단 정확도 10/10, **매뉴얼 정답률 7/9**, 평균 0.72초, 평균 도구 호출 2.8회, 평균 검수 시도 1.8회
- 틀린 문항: 5번(ess-overheat 대신 ess-soc-limit), 8번(dr-event 대신 peak-demand)
- 원인: 매뉴얼을 알람 문장의 **키워드 목록 순서**로만 고르고 질문 문장은 보지 않기 때문
  (`agent/graph.py`의 `_INCIDENT_TYPE_RULES`) → After에서 RAG + 우선순위 규칙으로 개선 예정
