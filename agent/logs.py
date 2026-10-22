"""[개선 I-12 — 강의 15장 Observability: 구조화 로그] 요청·노드 이벤트를 JSON 한 줄로 남긴다.

비구조화 로그("ERROR something failed near payment")는 grep으로 '추정'만 할 수 있지만, 필드가 정해진 JSON 로그는
`jq 'select(.event=="TaskFailed")'`처럼 필드로 바로 골라낼 수 있다. 필드 이름은 이 파일에서 고정한다
(서비스마다 user_id/uid처럼 이름이 달라지는 문제 방지):
  ts, level, event, thread_id, run_id(그래프 실행 ID), langsmith_run_id, node, elapsed_ms, code, message …

출력: 표준 에러(stderr). 환경변수 DER_LOG_FILE=logs/agent.jsonl을 주면 파일에도 쓴다. DER_LOG=off면 끈다.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger("der_ops")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   "level": record.levelname, **getattr(record, "fields", {})}
        return json.dumps(payload, ensure_ascii=False, default=str)


def _setup() -> None:
    if _logger.handlers:
        return
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    if os.environ.get("DER_LOG", "on") == "off":
        _logger.addHandler(logging.NullHandler())
        return
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(_JsonFormatter())
    _logger.addHandler(stream)
    if path := os.environ.get("DER_LOG_FILE"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(path, encoding="utf-8")
        file.setFormatter(_JsonFormatter())
        _logger.addHandler(file)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    _setup()
    _logger.log(level, event, extra={"fields": {"event": event, **fields}})


def log_graph_event(ev: dict, **ctx) -> None:
    """그래프 이벤트(dict)를 로그로. 실패 이벤트는 WARNING으로 올린다."""
    level = logging.WARNING if ev.get("type") in ("TaskFailed",) else logging.INFO
    fields = {k: v for k, v in ev.items() if k != "type" and not isinstance(v, (list, dict))}
    log_event(ev.get("type", "Event"), level, **ctx, **fields)
