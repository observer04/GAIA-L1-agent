from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.deps import get_agent_service
from backend.main import create_app
from backend.schemas.chat import ChatResponse, StreamEvent, TraceRecord, TraceTransition


class FakeAgentService:
    def run_chat(self, _request):
        return ChatResponse(
            run_id="run-1",
            session_id="sess-1",
            status="success",
            answer="42",
            submitted_answer="42",
            stop_reason="sufficient_evidence",
            latency_seconds=0.123,
            attempt_count=1,
            tool_trace=[{"tool": "web_search", "preview": "tool_call"}],
        )

    def stream_chat(self, _request):
        yield StreamEvent(
            event="run_started",
            run_id="run-1",
            session_id="sess-1",
            payload={"message_preview": "hello"},
        )
        yield StreamEvent(
            event="answer_delta",
            run_id="run-1",
            session_id="sess-1",
            payload={"delta": "4"},
        )
        yield StreamEvent(
            event="answer_delta",
            run_id="run-1",
            session_id="sess-1",
            payload={"delta": "2"},
        )
        yield StreamEvent(
            event="run_completed",
            run_id="run-1",
            session_id="sess-1",
            payload={"response": {"answer": "42"}},
        )

    def get_trace(self, run_id: str):
        if run_id != "run-1":
            return None
        return TraceRecord(
            run_id="run-1",
            session_id="sess-1",
            question="hello",
            status="success",
            stop_reason="sufficient_evidence",
            latency_seconds=0.123,
            attempt_count=1,
            tool_trace=[{"tool": "web_search", "preview": "tool_call"}],
            state_transitions=[
                TraceTransition(from_node="START", to_node="context_init"),
                TraceTransition(from_node="context_init", to_node="assistant"),
            ],
        )

    def list_traces(self, limit: int = 20):
        del limit
        record = self.get_trace("run-1")
        return [record] if record else []


def _test_client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    return TestClient(app)


def test_health_endpoint() -> None:
    client = _test_client()
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "timestamp" in payload


def test_security_headers_present() -> None:
    client = _test_client()
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cross-origin-opener-policy"] == "same-origin"


def test_chat_endpoint() -> None:
    client = _test_client()
    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-1"
    assert payload["answer"] == "42"


def test_chat_stream_endpoint() -> None:
    client = _test_client()
    response = client.post("/api/chat/stream", json={"message": "hello"})

    assert response.status_code == 200
    body = response.text
    assert "event: run_started" in body
    assert "event: answer_delta" in body
    assert "event: run_completed" in body


def test_trace_endpoints() -> None:
    client = _test_client()

    list_response = client.get("/api/traces")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    trace_response = client.get("/api/traces/run-1")
    assert trace_response.status_code == 200
    assert trace_response.json()["run_id"] == "run-1"

    missing_response = client.get("/api/traces/missing")
    assert missing_response.status_code == 404


def test_prefixed_base_path_routes(monkeypatch) -> None:
    monkeypatch.setenv("WEB_BASE_PATH", "/gaia_agent")

    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)

    prefixed_health = client.get("/gaia_agent/api/health")
    assert prefixed_health.status_code == 200

    unprefixed_health = client.get("/api/health")
    assert unprefixed_health.status_code == 404

    prefixed_api_root = client.get("/gaia_agent/api")
    assert prefixed_api_root.status_code == 200
    assert prefixed_api_root.json()["chat"] == "/gaia_agent/api/chat"

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["docs"] == "/gaia_agent/docs"


def test_prefixed_spa_serving(monkeypatch, tmp_path) -> None:
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)

    (dist_dir / "index.html").write_text("<html><body>gaia-spa</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('gaia');", encoding="utf-8")

    monkeypatch.setenv("WEB_BASE_PATH", "/gaia_agent")
    monkeypatch.setenv("WEB_STATIC_DIR", str(dist_dir))

    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)

    web_root = client.get("/gaia_agent/")
    assert web_root.status_code == 200
    assert "gaia-spa" in web_root.text

    asset = client.get("/gaia_agent/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text

    history_fallback = client.get("/gaia_agent/chat/123")
    assert history_fallback.status_code == 200
    assert "gaia-spa" in history_fallback.text


def test_rate_limit_applies_to_chat(monkeypatch) -> None:
    monkeypatch.setenv("WEB_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("WEB_RATE_LIMIT_REQUESTS_PER_MINUTE", "1")
    monkeypatch.setenv("WEB_RATE_LIMIT_WINDOW_SECONDS", "60")

    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)

    first = client.post("/api/chat", json={"message": "hello"})
    second = client.post("/api/chat", json={"message": "hello again"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1
    assert second.json()["detail"].startswith("Rate limit exceeded")
