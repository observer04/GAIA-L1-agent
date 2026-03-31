from __future__ import annotations

from ..services.agent_service import AgentService

_agent_service = AgentService()


def get_agent_service() -> AgentService:
    return _agent_service
