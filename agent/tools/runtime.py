from __future__ import annotations

from contextvars import ContextVar, Token

_RUNTIME_CONTEXT: ContextVar[dict[str, str]] = ContextVar(
    "gaia_tool_runtime_context", default={}
)


def set_runtime_context(task_id: str, working_dir: str) -> Token:
    return _RUNTIME_CONTEXT.set({"task_id": task_id or "", "working_dir": working_dir or ""})


def reset_runtime_context(token: Token) -> None:
    _RUNTIME_CONTEXT.reset(token)


def get_runtime_context() -> dict[str, str]:
    return _RUNTIME_CONTEXT.get()


def get_runtime_task_id() -> str:
    return get_runtime_context().get("task_id", "")


def get_runtime_working_dir() -> str:
    return get_runtime_context().get("working_dir", "")
