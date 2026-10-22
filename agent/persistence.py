"""[개선 I-3 — 강의 13장 Checkpointer Factory] 그래프 상태 저장소(체크포인터)를 만드는 곳을 이 파일 하나로 모은다.

Before(after 태그)는 build_graph() 안에서 InMemorySaver()를 직접 만들어서, 승인 대기(interrupt)로 멈춘 그래프가
서버 재시작과 함께 사라졌다. 이제 server.py의 lifespan이 이 팩토리로 체크포인터를 만들어 그래프에 넣어 주고,
그래프·테스트 코드는 체크포인터를 "받기만" 한다 → 저장소를 바꿀 때 이 파일만 고치면 된다.

종류(환경변수 CHECKPOINTER)
- sqlite (기본) : checkpoints.db 파일에 저장 — 서버를 재시작해도 승인 대기가 이어진다
- memory        : 프로세스 메모리 — 테스트·측정용 (재시작하면 사라짐)

강의는 PostgreSQL을 채택했다(재시작 내구성 + 여러 서버 인스턴스 간 공유). 이 프로젝트는 서버 1대로 로컬 운영하므로
SQLite로 재시작 내구성만 확보하고, 서버를 여러 대로 늘릴 때 Postgres로 바꾼다 — doc/설계_결정_기록.md ADR-0005 참고.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "checkpoints.db"


@asynccontextmanager
async def create_checkpointer(kind: str | None = None, path: Path | str | None = None
                              ) -> AsyncIterator[BaseCheckpointSaver]:
    kind = kind or os.environ.get("CHECKPOINTER", "sqlite")
    if kind == "memory":
        yield InMemorySaver()
        return
    if kind != "sqlite":
        raise RuntimeError(f"알 수 없는 CHECKPOINTER={kind!r} (sqlite 또는 memory)")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    db = Path(path or os.environ.get("CHECKPOINT_DB", DEFAULT_DB))
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        await saver.setup()   # 테이블 준비 (여러 번 불러도 안전)
        yield saver


async def forget_run(checkpointer: BaseCheckpointSaver, run_id: str) -> None:
    """끝난 실행의 체크포인트를 지운다 — 질문마다 run_id로 쌓이기만 하면 저장소가 계속 커진다."""
    try:
        await checkpointer.adelete_thread(run_id)
    except Exception:
        pass   # 정리 실패가 사용자 응답을 막지 않게
