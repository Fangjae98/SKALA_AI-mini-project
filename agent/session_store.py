"""Wave 12 — 프로세스 밖(SQLite 파일)에 두는 세션 저장소.

서버가 재시작돼도(새 프로세스, 새 인스턴스) 같은 session.db 파일을 다시 열므로
대화 이력이 유지된다. in-memory dict였다면 재시작 즉시 소실됐을 상태다.
"""
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("SESSION_DB", Path(__file__).parent.parent / "session.db"))


class SqliteSessionStore:
    def __init__(self, db_path: Path | None = None):
        db_path = db_path or Path(os.environ.get("SESSION_DB", DB_PATH))
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " thread_id TEXT, role TEXT, content TEXT, created TEXT)"
        )
        self.conn.commit()

    def load(self, thread_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE thread_id=? ORDER BY rowid",
            (thread_id,),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def append(self, thread_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?)",
            (thread_id, role, content, datetime.now().isoformat()),
        )
        self.conn.commit()

    # ── [개선 I-3] 승인 대기 목록도 같은 파일에 저장 — 서버를 재시작해도 어떤 실행이 승인을 기다리는지 기억한다.
    def _ensure_pending_table(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_approvals ("
            " thread_id TEXT PRIMARY KEY, run_id TEXT, message TEXT, created TEXT)"
        )

    def set_pending(self, thread_id: str, run_id: str, message: str) -> None:
        self._ensure_pending_table()
        self.conn.execute("INSERT OR REPLACE INTO pending_approvals VALUES (?,?,?,?)",
                          (thread_id, run_id, message, datetime.now().isoformat()))
        self.conn.commit()

    def get_pending(self, thread_id: str) -> dict | None:
        self._ensure_pending_table()
        row = self.conn.execute("SELECT run_id, message FROM pending_approvals WHERE thread_id=?",
                                (thread_id,)).fetchone()
        return {"run_id": row[0], "message": row[1]} if row else None

    def pop_pending(self, thread_id: str) -> dict | None:
        pend = self.get_pending(thread_id)
        if pend:
            self.conn.execute("DELETE FROM pending_approvals WHERE thread_id=?", (thread_id,))
            self.conn.commit()
        return pend

    def close(self) -> None:
        self.conn.close()
