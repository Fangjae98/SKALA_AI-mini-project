"""Wave 9 — Transient 에러만 exponential backoff로 재시도.

asyncio.TimeoutError(일시적)만 재시도하고, 그 외(ValueError 등 영구적)는 즉시 실패로 확정한다.

[개선 I-1 — 강의 10장 Error Handling] Base는 재시도가 끝나거나 영구 오류가 나면 원래 예외를 그대로 던져
그래프 전체가 멈췄다. 이제 실패를 ToolError(code, user_message) 하나로 표준화해서, 호출한 쪽(전문 에이전트)이
"그 설비만 실패"로 기록하고 나머지 조사를 계속할 수 있게 한다(Error 격리 + Partial Result).
재시도 횟수·대기 시간은 config/rules.json의 retry에서 읽는다(강의 기준: 최대 2회 재시도, 0.5 → 1.5초).
"""
import asyncio

from langsmith import traceable

from .config import RULES

TOOL_TIMEOUT_AFTER_RETRY = "TOOL_TIMEOUT_AFTER_RETRY"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"


class ToolError(Exception):
    """MCP 도구 호출 실패를 표준 형식으로 담는다. code는 화면·로그용, user_message는 보고서에 넣을 문장."""

    def __init__(self, code: str, user_message: str, tool_name: str = "", retries: int = 0):
        super().__init__(f"[{code}] {tool_name}: {user_message}")
        self.code = code
        self.user_message = user_message
        self.tool_name = tool_name
        self.retries = retries

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.user_message, "tool": self.tool_name,
                "retries": self.retries}


@traceable(name="with_retry")
async def with_retry(fn, *args, retries: int | None = None, base_delay: float | None = None,
                     tool_name: str | None = None, **kwargs):
    """fn(*args, **kwargs)를 실행하고 (결과, 재시도 횟수)를 돌려준다. 실패하면 ToolError를 던진다."""
    cfg = RULES["retry"]
    max_retries = cfg["max_retries"] if retries is None else retries
    delay = cfg["initial_delay_s"] if base_delay is None else base_delay
    name = tool_name or (args[0] if args and isinstance(args[0], str) else getattr(fn, "__name__", "tool"))
    attempt = 0
    while True:
        try:
            return await fn(*args, **kwargs), attempt
        except asyncio.TimeoutError as e:
            if attempt >= max_retries:
                raise ToolError(TOOL_TIMEOUT_AFTER_RETRY,
                                f"응답 시간 초과 — {attempt}회 재시도 후에도 응답이 없습니다", name, attempt) from e
            await asyncio.sleep(delay)
            delay *= cfg["backoff"]
            attempt += 1
        except ToolError:
            raise
        except Exception as e:   # 영구 오류 — 재시도해도 같은 결과이므로 바로 실패 확정
            raise ToolError(TOOL_EXECUTION_ERROR, f"조회 실패({type(e).__name__}: {e})", name, attempt) from e
