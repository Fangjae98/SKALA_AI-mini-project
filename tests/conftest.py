"""pytest 공통 설정 — 프로젝트 폴더를 import 경로에 넣고, LangSmith 기록과 실제 LLM 호출을 끈다."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LLM_BACKEND"] = "fake"
os.environ.pop("EMBEDDING_BACKEND", None)
