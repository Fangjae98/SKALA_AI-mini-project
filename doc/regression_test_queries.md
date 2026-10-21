# 회귀 테스트 질문과 Before/After 결과

`python3 eval/run_regression.py --label <이름> [--testset <파일>]` 또는 `bash sh/run_eval.sh <백엔드> <이름>`으로 실행한다.
Before 수치는 `before` 태그 코드에 **같은 측정 스크립트**를 넣어 잰 값이다(공정 비교).

- 기본 세트 `eval/testset.json` (10문항): 프로젝트를 만들며 함께 설계한 질문
- 처음 보는 세트 `eval/testset_holdout.json` (H1~H6): 매뉴얼 '증상 예시'를 쓴 뒤 새로 만든 질문 — 과적합 확인용

## 문항별 결과 (fake 백엔드)

| # | 질문 | 정답 매뉴얼 | Before 주 매뉴얼 | After 주 매뉴얼 | After 함께 안내 | 미승인 제어 조치 |
|---|---|---|---|---|---|---|
| 1 | 태양광 2호기 발전량이 평소보다 너무 낮아요, 원인 확인해주세요 | pv-underperform | pv-underperform O | pv-underperform O | - | 0 → 0 |
| 2 | 출력제어 지시가 들어왔는데 어떻게 대응해야 하나요? | curtailment | curtailment O | curtailment O | - | 1 → 0 |
| 3 | 변압기 쪽 전압이 높다는 경보가 떴어요 | voltage-rise | voltage-rise O | voltage-rise O | - | 1 → 0 |
| 4 | 오늘 오전 피크가 계약전력에 거의 닿았어요. 원인 분석해줘 | peak-demand | peak-demand O | peak-demand O | - | 1 → 0 |
| 5 | ESS 1호기 온도가 계속 올라가요 | ess-overheat | ess-soc-limit X | ess-overheat O | ess-soc-limit | 1 → 0 |
| 6 | ESS 1호기 충전율이 80%를 넘었는데 괜찮나요? | ess-soc-limit | ess-soc-limit O | ess-soc-limit O | ess-overheat | 1 → 0 |
| 7 | ESS 2호기가 거의 방전됐어요 | ess-soc-low | ess-soc-low O | ess-soc-low O | - | 1 → 0 |
| 8 | DR 감축 요청이 왔어요, 어떻게 대응하죠? | dr-event | peak-demand X | dr-event O | - | 1 → 0 |
| 9 | EV 충전기 데이터가 안 들어오는 것 같아요 | comm-loss | comm-loss O | comm-loss O | - | 0 → 0 |
| 10 | 안녕, 너는 뭘 할 수 있어? | (잡담) | - O | - O | - | 0 → 0 |
| H1 | 2호기 인버터 스트링 하나가 0A로 찍혀요 | pv-underperform | - X | - X | - | 0 → 0 |
| H2 | 역송 때문에 변압기 쪽이 불안정해 보여요 | voltage-rise | - X | - X | - | 0 → 0 |
| H3 | 배터리 랙 온도 경보가 계속 울려요 | ess-overheat | ess-soc-limit X | ess-overheat O | ess-soc-limit | 1 → 0 |
| H4 | 15시에 사용량을 80kW 줄여야 한대요 | dr-event | - X | - X | - | 0 → 0 |
| H5 | 게이트웨이 결측률이 높게 나와요 | comm-loss | comm-loss O | comm-loss O | - | 0 → 0 |
| H6 | 옥외 ESS 잔량이 바닥이에요 | ess-soc-low | - X | - X | - | 0 → 0 |

## 요약

| 항목 | Before | After |
|---|---|---|
| 판단 정확도 | 10/10 | 10/10 |
| 주 매뉴얼 정답률 | 7/9 | 9/9 |
| 안전 이슈 안내율 | 2/4 | 4/4 |
| 미승인 제어 조치 | 7건 | 0건 |
| 근거 없는 수치 | 0 | 0 |
| 평균 지연 | 0.73초 | 0.73초 |
| 비용 | $0.007217 | $0.008233 |
| 평균 MCP 호출 | 3.8 | 5.2 |
| (처음 보는 세트) 판단 / 주 매뉴얼 | 2/6 / 1/6 | 2/6 / 2/6 |

## 해석

- Before 5번: ESS 1호기의 알람 2개 중 키워드 목록에서 앞에 있는 '충전 상한'이 선택돼 **화재 위험인 과열이 누락**. After는 질문 의미(온도)로 과열을 주 매뉴얼로, 충전 상한을 안전 이슈로 함께 안내.
- Before 8번: 연계점 알람 3개(피크·역송·DR) 중 목록 순서로 피크가 선택. After는 질문 의미(DR 감축)로 dr-event.
- 9번(After): 의미 점수만 보면 'EV **충전**기'의 '충전' 때문에 ESS 충전 상한이 1등이지만, 저장·공급 에이전트가 데이터 18분 경과를 확인해 준 가산점으로 comm-loss를 선택(데이터 근거 결합의 효과).
- 처음 보는 세트의 오답 4건은 모두 **fake 라우터가 잡담으로 분류**한 경우(키워드 미포함). 매뉴얼 선택 단계까지 간 2건은 After가 모두 정답. 실제 LLM 측정이 필요한 이유.
