"""pytest 공통 설정 — 프로젝트 폴더를 import 경로에 넣고, LangSmith 기록과 실제 LLM 호출을 끈다."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LLM_BACKEND"] = "fake"
os.environ.pop("EMBEDDING_BACKEND", None)
os.environ.pop("DER_FAULT_INJECT", None)
os.environ["DER_LOG"] = "off"
# 테스트는 정답이 고정된 '전체 이상' 시나리오로 돈다(화면 시연용 data/monitoring.json은 수시로 바뀔 수 있으므로)
os.environ["DER_MONITORING_FILE"] = "data/scenarios/full_incident.json"          # 구조화 로그는 테스트 출력에서 끈다
os.environ["CHECKPOINTER"] = "memory"  # 서버 테스트가 실제 checkpoints.db를 건드리지 않게
