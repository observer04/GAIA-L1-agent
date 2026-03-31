from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...schemas.chat import TraceRecord
from ...services.agent_service import AgentService
from ..deps import get_agent_service

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("", response_model=list[TraceRecord])
def list_trace_runs(
    limit: int = Query(default=20, ge=1, le=200),
    service: AgentService = Depends(get_agent_service),
) -> list[TraceRecord]:
    return service.list_traces(limit=limit)


@router.get("/{run_id}", response_model=TraceRecord)
def get_trace_run(run_id: str, service: AgentService = Depends(get_agent_service)) -> TraceRecord:
    record = service.get_trace(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace run not found: {run_id}")
    return record
