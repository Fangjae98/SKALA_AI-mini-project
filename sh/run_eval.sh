#!/usr/bin/env bash
#
# der_ops_agent — 회귀 테스트(Before/After 측정)를 한 번에 돌리는 스크립트. [After 추가]
#
# 사용법:
#   bash sh/run_eval.sh                 # fake, 결과 이름 after
#   bash sh/run_eval.sh ollama after    # Ollama 실모델로 측정 (ollama serve와 모델 3개 필요)
#   bash sh/run_eval.sh openai after    # OpenAI로 측정 (과금, .env에 OPENAI_API_KEY 필요)
#
# 기본 질문 10개(eval/testset.json)와 처음 보는 질문 6개(eval/testset_holdout.json)를 모두 돌리고,
# 결과는 eval/results/<이름>_<백엔드>_<세트>.json에 저장된다.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x ".venv/bin/python3" ] && export PATH="$PWD/.venv/bin:$PATH"
[ -f .env ] && { set -a; source .env; set +a; }

BACKEND="${1:-fake}"
LABEL="${2:-after}"
export LLM_BACKEND="$BACKEND"
export LANGCHAIN_PROJECT="der-ops-agent-eval-$BACKEND"

if [ "$BACKEND" = "openai" ]; then
    read -r -p "OpenAI API를 사용합니다(과금). 계속할까요? [y/N] " ans
    [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "취소했습니다."; exit 0; }
fi

python3 eval/run_regression.py --label "$LABEL" --testset eval/testset.json
python3 eval/run_regression.py --label "$LABEL" --testset eval/testset_holdout.json
