from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import AgentConfig
from .postprocess import has_final_answer_marker, is_unusable_answer, normalize_answer
from .prompts import REPLAN_USER_PROMPT, SYSTEM_PROMPT
from .state import GaiaAgentState
from .tools import download_task_file_raw, get_tools
from .tools.runtime import reset_runtime_context, set_runtime_context


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _extract_message_content(message: Any) -> str:
    return _extract_content(getattr(message, "content", ""))


def _latest_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


class GaiaLangGraphAgent:
    """Production-style GAIA agent with staged LangGraph workflow and robust fallbacks."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()
        self.config.assert_required()

        self.tools = get_tools()
        self.tool_node = ToolNode(self.tools)
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            google_api_key=self.config.google_api_key,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _context_init_node(self, state: GaiaAgentState) -> GaiaAgentState:
        task_id = (state.get("task_id") or "").strip()
        work_dir = Path(self.config.local_working_dir) / (task_id or "unknown")
        work_dir.mkdir(parents=True, exist_ok=True)

        return {
            "working_dir": str(work_dir),
            "attempt_count": state.get("attempt_count", 0),
            "downloaded_files": state.get("downloaded_files", []),
            "tool_trace": state.get("tool_trace", []),
            "last_error": state.get("last_error", ""),
            "stop_reason": state.get("stop_reason", ""),
        }

    def _task_file_fetch_node(self, state: GaiaAgentState) -> GaiaAgentState:
        task_id = (state.get("task_id") or "").strip()
        if not task_id:
            return {}

        existing = list(state.get("downloaded_files", []))
        if existing:
            return {}

        result = download_task_file_raw(
            task_id=task_id,
            api_url=self.config.api_url,
            destination_dir=state.get("working_dir") or None,
        )

        status = result.get("status")
        if status == "downloaded":
            downloaded_path = str(result.get("path", "")).strip()
            return {
                "downloaded_files": [downloaded_path],
                "messages": [
                    HumanMessage(
                        content=(
                            "[System note: attached task file downloaded to "
                            f"{downloaded_path}. Use file tools if needed.]"
                        )
                    )
                ],
            }

        if status == "no_file":
            return {
                "messages": [
                    HumanMessage(content="[System note: no attached file for this task_id.]")
                ]
            }

        error_message = str(result.get("message", "task download failed"))
        return {
            "last_error": error_message,
            "messages": [
                HumanMessage(
                    content=(
                        "[System note: automatic task file prefetch failed. "
                        f"Error: {error_message}]"
                    )
                )
            ],
        }

    def _assistant_node(self, state: GaiaAgentState) -> GaiaAgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.get("messages", []))
        attempt_count = int(state.get("attempt_count", 0)) + 1

        try:
            ai_message = self.llm_with_tools.invoke(messages)
        except Exception as err:  # noqa: BLE001
            ai_message = AIMessage(content=f"ERROR: model invocation failed ({err})")

        tool_trace = list(state.get("tool_trace", []))
        for call in getattr(ai_message, "tool_calls", []):
            if isinstance(call, dict):
                tool_trace.append({"tool": call.get("name", "unknown"), "preview": "tool_call"})

        return {
            "messages": [ai_message],
            "attempt_count": attempt_count,
            "tool_trace": tool_trace,
        }

    def _evidence_check_node(self, state: GaiaAgentState) -> GaiaAgentState:
        attempts = int(state.get("attempt_count", 0))
        messages = state.get("messages", [])
        last_error = state.get("last_error", "")

        tool_trace = list(state.get("tool_trace", []))
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                preview = _extract_message_content(msg)
                tool_trace.append(
                    {
                        "tool": getattr(msg, "name", "unknown"),
                        "preview": preview[:400],
                    }
                )
                if "ERROR:" in preview.upper():
                    last_error = preview[:500]
                if len(tool_trace) >= 12:
                    break
        tool_trace = tool_trace[-12:]

        last_ai = _latest_ai_message(messages)
        raw_output = _extract_message_content(last_ai) if last_ai else ""
        has_final_marker = has_final_answer_marker(raw_output)
        candidate = normalize_answer(raw_output) if has_final_marker else ""

        if candidate and not is_unusable_answer(candidate):
            return {
                "candidate_answer": candidate,
                "stop_reason": "sufficient_evidence",
                "last_error": last_error,
                "tool_trace": tool_trace,
            }

        if attempts >= self.config.max_iterations:
            return {
                "candidate_answer": "I don't know",
                "stop_reason": "max_iterations_reached",
                "last_error": last_error,
                "tool_trace": tool_trace,
            }

        return {
            "stop_reason": "insufficient_evidence",
            "last_error": last_error,
            "tool_trace": tool_trace,
        }

    @staticmethod
    def _replan_node(state: GaiaAgentState) -> GaiaAgentState:
        _ = state
        return {"messages": [HumanMessage(content=REPLAN_USER_PROMPT)]}

    @staticmethod
    def _finalize_node(state: GaiaAgentState) -> GaiaAgentState:
        candidate = state.get("candidate_answer") or "I don't know"
        stop_reason = state.get("stop_reason") or "completed"

        return {
            "candidate_answer": candidate,
            "messages": [AIMessage(content=f"FINAL ANSWER: {candidate}")],
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _route_after_assistant(state: GaiaAgentState) -> str:
        messages = state.get("messages", [])
        last_ai = _latest_ai_message(messages)
        if not last_ai:
            return "evidence_check"

        tool_calls = getattr(last_ai, "tool_calls", None) or []
        if tool_calls:
            return "tools"

        return "evidence_check"

    def _route_after_evidence(self, state: GaiaAgentState) -> str:
        if state.get("candidate_answer"):
            return "finalize"

        if int(state.get("attempt_count", 0)) >= self.config.max_iterations:
            return "finalize"

        return "replan"

    def _build_graph(self):
        builder = StateGraph(GaiaAgentState)

        builder.add_node("context_init", self._context_init_node)
        builder.add_node("task_file_fetch", self._task_file_fetch_node)
        builder.add_node("assistant", self._assistant_node)
        builder.add_node("tools", self.tool_node)
        builder.add_node("evidence_check", self._evidence_check_node)
        builder.add_node("replan", self._replan_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "context_init")
        builder.add_edge("context_init", "task_file_fetch")
        builder.add_edge("task_file_fetch", "assistant")

        builder.add_conditional_edges(
            "assistant",
            self._route_after_assistant,
            {
                "tools": "tools",
                "evidence_check": "evidence_check",
            },
        )

        builder.add_edge("tools", "evidence_check")
        builder.add_conditional_edges(
            "evidence_check",
            self._route_after_evidence,
            {
                "replan": "replan",
                "finalize": "finalize",
            },
        )
        builder.add_edge("replan", "assistant")

        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def _latest_raw_output(result_state: dict[str, Any]) -> str:
        messages = result_state.get("messages", [])
        if not messages:
            return ""
        return _extract_message_content(messages[-1])

    def run_task(self, question: str, task_id: str, run_label: str = "local") -> dict[str, Any]:
        started = time.perf_counter()
        safe_task_id = (task_id or "").strip()
        run_dir = Path(self.config.local_working_dir) / (safe_task_id or "task_unknown")
        run_dir.mkdir(parents=True, exist_ok=True)

        token = set_runtime_context(task_id=safe_task_id, working_dir=str(run_dir))
        prompt = f"Question: {question}\ntask_id: {safe_task_id}"

        initial_state: GaiaAgentState = {
            "question": question,
            "task_id": safe_task_id,
            "working_dir": str(run_dir),
            "messages": [HumanMessage(content=prompt)],
            "attempt_count": 0,
            "downloaded_files": [],
            "tool_trace": [],
            "last_error": "",
            "candidate_answer": "",
            "stop_reason": "",
        }

        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self.graph.invoke,
                initial_state,
                {
                    "recursion_limit": self.config.recursion_limit,
                    "tags": [f"gaia:{run_label}", f"task:{safe_task_id or 'unknown'}"],
                    "metadata": {
                        "task_id": safe_task_id,
                        "run_label": run_label,
                        "langsmith_project": self.config.langsmith_project,
                    },
                },
            )
            result = future.result(timeout=self.config.timeout_seconds)

            candidate = str(result.get("candidate_answer", "")).strip()
            if not candidate:
                messages = result.get("messages", [])
                last_content = _extract_message_content(messages[-1]) if messages else ""
                candidate = normalize_answer(last_content)

            return {
                "status": "success",
                "task_id": safe_task_id,
                "submitted_answer": candidate,
                "raw_output": self._latest_raw_output(result),
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempt_count": int(result.get("attempt_count", 0)),
                "stop_reason": result.get("stop_reason", ""),
                "downloaded_files": result.get("downloaded_files", []),
                "tool_trace": result.get("tool_trace", []),
            }
        except concurrent.futures.TimeoutError:
            future.cancel()
            return {
                "status": "success",
                "task_id": safe_task_id,
                "submitted_answer": "I don't know",
                "raw_output": "FINAL ANSWER: I don't know",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempt_count": 0,
                "stop_reason": "timeout",
                "downloaded_files": [],
                "tool_trace": [],
            }
        except Exception as err:  # noqa: BLE001
            return {
                "status": "error",
                "task_id": safe_task_id,
                "submitted_answer": "I don't know",
                "raw_output": f"ERROR: {err}",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempt_count": 0,
                "stop_reason": "exception",
                "downloaded_files": [],
                "tool_trace": [],
            }
        finally:
            if "executor" in locals():
                executor.shutdown(wait=False, cancel_futures=True)
            reset_runtime_context(token)

    def __call__(self, question: str, task_id: str) -> str:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.run_task, question, task_id, "submission")
                result = future.result(timeout=self.config.timeout_seconds)
                return str(result.get("submitted_answer", "I don't know"))
        except concurrent.futures.TimeoutError:
            return "I don't know"
        except Exception:  # noqa: BLE001
            return "I don't know"


def build_graph_for_studio():
    """LangGraph Studio entrypoint."""
    return GaiaLangGraphAgent().graph
