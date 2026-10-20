#!/usr/bin/env bash
#
# der_ops_agent — 가짜 LLM(fake, 결정론·무료·인터넷 불필요)으로 에이전트를 기동만 해두는 스크립트.
#
# [Base 대비 추가] Base 학생 배포본에는 fake 모드 실행 스크립트가 없었다. Ollama 모델을 아직
# 안 받았거나, 코드 흐름만 빠르게 확인하고 싶을 때 이걸로 띄운다(항상 같은 답이 나와서
# Before/After 비교와 테스트에 쓰기 좋다).
#
# .env가 없어도 동작한다(LangSmith 기록만 꺼짐). 종료하려면 이 터미널에서 Ctrl+C.
#
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x ".venv/bin/python3" ] && export PATH="$PWD/.venv/bin:$PATH"

if [ -f .env ]; then
    set -a; source .env; set +a
    export LANGCHAIN_PROJECT="der-ops-agent-fake"
else
    export LANGCHAIN_TRACING_V2=false
    echo ".env가 없어 LangSmith 기록 없이 실행합니다."
fi

lsof -ti:8100 | xargs kill -9 2>/dev/null || true
rm -f session.db

cleanup() {
    echo
    echo "에이전트를 종료합니다..."
    lsof -ti:8100 | xargs kill -9 2>/dev/null || true
}
trap cleanup EXIT

LLM_BACKEND=fake SUPERVISOR_VERSION=v2 python3 server.py &
SERVER_PID=$!
for _ in $(seq 1 40); do
    curl -s http://127.0.0.1:8100/health >/dev/null 2>&1 && break
    sleep 0.5
done

echo
curl -s http://127.0.0.1:8100/health | python3 -m json.tool --no-ensure-ascii

cat <<EOF

################################################################
 에이전트가 떠 있습니다 (http://127.0.0.1:8100, LLM_BACKEND=fake)
################################################################

1) curl로 말 걸어보기
   curl -s -X POST localhost:8100/chat -H 'Content-Type: application/json' \\
     -d '{"message":"태양광 2호기 발전량이 평소보다 너무 낮아요, 원인 확인해주세요","thread_id":"t-test"}' \\
     | python3 -m json.tool --no-ensure-ascii

2) 브라우저 채팅 화면 (새 터미널에서)
   bash sh/run_frontend.sh  → http://localhost:8600

3) 회귀 테스트(Before/After 측정)
   python3 eval/run_regression.py

종료하려면 이 터미널에서 Ctrl+C.
EOF

wait "$SERVER_PID"
