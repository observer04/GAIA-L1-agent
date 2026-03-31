from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

from agent import GaiaLangGraphAgent
from agent.config import AgentConfig

from ..schemas.chat import ChatRequest, ChatResponse, StreamEvent, TraceRecord, TraceTransition


class AgentService:
    """Service boundary around GaiaLangGraphAgent for API use."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()
        self._agent_lock = Lock()
        self._agent: GaiaLangGraphAgent | None = None

        self._trace_lock = Lock()
        self._trace_store: OrderedDict[str, TraceRecord] = OrderedDict()
        self._trace_store_max = max(20, int(self.config.trace_store_max_runs))

    def _get_agent(self) -> GaiaLangGraphAgent:
        with self._agent_lock:
            if self._agent is None:
                self._agent = GaiaLangGraphAgent(config=self.config)
            return self._agent

    @staticmethod
    def _session_id_or_new(session_id: str | None) -> str:
        cleaned = str(session_id or "").strip()
        return cleaned or f"sess-{uuid4().hex}"

    @staticmethod
    def _task_id_or_new(task_id: str | None) -> str:
        cleaned = str(task_id or "").strip()
        return cleaned or f"chat-{uuid4().hex}"

    @staticmethod
    def _default_state_transitions(tool_trace: list[dict[str, Any]]) -> list[TraceTransition]:
        transitions = [
            TraceTransition(from_node="START", to_node="context_init"),
            TraceTransition(from_node="context_init", to_node="task_file_fetch"),
            TraceTransition(from_node="task_file_fetch", to_node="assistant"),
        ]

        if tool_trace:
            transitions.append(TraceTransition(from_node="assistant", to_node="tools"))
            transitions.append(TraceTransition(from_node="tools", to_node="evidence_check"))

        transitions.append(TraceTransition(from_node="assistant", to_node="evidence_check"))
        transitions.append(TraceTransition(from_node="evidence_check", to_node="finalize"))
        return transitions

    def _store_trace(self, record: TraceRecord) -> None:
        with self._trace_lock:
            self._trace_store[record.run_id] = record
            self._trace_store.move_to_end(record.run_id)

            while len(self._trace_store) > self._trace_store_max:
                self._trace_store.popitem(last=False)

    def _run_chat_internal(
        self,
        request: ChatRequest,
        *,
        run_id: str,
        session_id: str,
    ) -> tuple[ChatResponse, TraceRecord]:
        message = str(request.message or "").strip()
        if not message:
            raise ValueError("message is empty")

        max_prompt = max(512, int(self.config.web_max_prompt_chars))
        if len(message) > max_prompt:
            message = message[:max_prompt]

        task_id = self._task_id_or_new(request.task_id)
        run_label = str(request.run_label or "web-chat").strip() or "web-chat"

        run_result = self._get_agent().run_task(
            question=message,
            task_id=task_id,
            run_label=run_label,
            level=str(request.level or "").strip(),
            file_name=str(request.file_name or "").strip(),
        )

        submitted_answer = str(run_result.get("submitted_answer", "I don't know") or "I don't know").strip()
        if not submitted_answer:
            submitted_answer = "I don't know"

        tool_trace = run_result.get("tool_trace", []) or []
        if not isinstance(tool_trace, list):
            tool_trace = []

        response = ChatResponse(
            run_id=run_id,
            session_id=session_id,
            status=str(run_result.get("status", "error")),
            answer=submitted_answer,
            submitted_answer=submitted_answer,
            stop_reason=str(run_result.get("stop_reason", "")),
            latency_seconds=float(run_result.get("latency_seconds", 0.0) or 0.0),
            attempt_count=int(run_result.get("attempt_count", 0) or 0),
            tool_trace=tool_trace,
        )

        trace_record = TraceRecord(
            run_id=run_id,
            session_id=session_id,
            question=message,
            status=response.status,
            stop_reason=response.stop_reason,
            latency_seconds=response.latency_seconds,
            attempt_count=response.attempt_count,
            tool_trace=response.tool_trace,
            state_transitions=self._default_state_transitions(response.tool_trace),
        )
        self._store_trace(trace_record)
        return response, trace_record

    def run_chat(self, request: ChatRequest) -> ChatResponse:
        run_id = f"run-{uuid4().hex}"
        session_id = self._session_id_or_new(request.session_id)
        response, _ = self._run_chat_internal(request, run_id=run_id, session_id=session_id)
        return response

    def stream_chat(self, request: ChatRequest) -> Iterator[StreamEvent]:
        run_id = f"run-{uuid4().hex}"
        session_id = self._session_id_or_new(request.session_id)

        yield StreamEvent(
            event="run_started",
            run_id=run_id,
            session_id=session_id,
            payload={"message_preview": str(request.message)[:120]},
        )

        try:
            response, trace_record = self._run_chat_internal(
                request,
                run_id=run_id,
                session_id=session_id,
            )
        except Exception as err:  # noqa: BLE001
            yield StreamEvent(
                event="error",
                run_id=run_id,
                session_id=session_id,
                payload={"detail": f"{err.__class__.__name__}: {err}"},
            )
            return

        for idx, trace_item in enumerate(trace_record.tool_trace):
            yield StreamEvent(
                event="tool_trace",
                run_id=run_id,
                session_id=session_id,
                payload={"index": idx, "trace": trace_item},
            )

        answer = str(response.answer or "")
        if answer:
            chunk_size = max(8, int(self.config.stream_answer_chunk_chars))
            for start in range(0, len(answer), chunk_size):
                chunk = answer[start : start + chunk_size]
                yield StreamEvent(
                    event="answer_delta",
                    run_id=run_id,
                    session_id=session_id,
                    payload={
                        "delta": chunk,
                        "cursor": start + len(chunk),
                        "total": len(answer),
                    },
                )

        yield StreamEvent(
            event="run_completed",
            run_id=run_id,
            session_id=session_id,
            payload={"response": response.model_dump()},
        )

    def get_trace(self, run_id: str) -> TraceRecord | None:
        with self._trace_lock:
            record = self._trace_store.get(str(run_id or "").strip())
            if record is None:
                return None
            return record.model_copy(deep=True)

    def list_traces(self, limit: int = 20) -> list[TraceRecord]:
        safe_limit = max(1, min(int(limit), 200))
        with self._trace_lock:
            records = list(self._trace_store.values())[-safe_limit:]
        return [record.model_copy(deep=True) for record in reversed(records)]
