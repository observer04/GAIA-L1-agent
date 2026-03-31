from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=24000)
    session_id: str | None = Field(default=None, max_length=128)
    run_label: str = Field(default="web-chat", max_length=64)
    task_id: str | None = Field(default=None, max_length=128)
    level: str = Field(default="", max_length=16)
    file_name: str = Field(default="", max_length=256)


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    status: str
    answer: str
    submitted_answer: str
    stop_reason: str = ""
    latency_seconds: float = 0.0
    attempt_count: int = 0
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class TraceTransition(BaseModel):
    from_node: str
    to_node: str


class TraceRecord(BaseModel):
    run_id: str
    session_id: str
    question: str
    created_at: str = Field(default_factory=utc_now_iso)
    status: str = "success"
    stop_reason: str = ""
    latency_seconds: float = 0.0
    attempt_count: int = 0
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    state_transitions: list[TraceTransition] = Field(default_factory=list)


class StreamEvent(BaseModel):
    event: Literal["run_started", "tool_trace", "answer_delta", "run_completed", "error"]
    run_id: str
    session_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)
