from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...schemas.chat import ChatRequest, ChatResponse, StreamEvent
from ...services.agent_service import AgentService
from ..deps import get_agent_service

router = APIRouter(prefix="/chat", tags=["chat"])


def _serialize_sse(event: StreamEvent) -> str:
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"


@router.post("", response_model=ChatResponse)
def chat_completion(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    try:
        return service.run_chat(request)
    except ValueError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"chat execution failed ({err.__class__.__name__}: {err})",
        ) from err


@router.post("/stream")
def chat_completion_stream(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    def event_stream():
        for event in service.stream_chat(request):
            yield _serialize_sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
