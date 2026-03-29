from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GaiaAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    task_id: str
    level: str
    file_name: str
    downloaded_files: list[str]
    task_file_status: str
    task_file_error: str
    working_dir: str
    tool_trace: list[dict[str, Any]]
    search_history: list[dict[str, Any]]
    latest_search_observation: dict[str, Any]
    attempt_count: int
    max_iterations_for_task: int
    last_error: str
    candidate_answer: str
    stop_reason: str
