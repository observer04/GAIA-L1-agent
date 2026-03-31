from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Final

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.api.routes.chat import router as chat_router
from backend.api.routes.health import router as health_router
from backend.api.routes.traces import router as traces_router


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class _FixedWindowRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._buckets: dict[str, list[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = monotonic()
        with self._lock:
            bucket = self._buckets.get(key, [])
            cutoff = now - self.window_seconds
            bucket = [ts for ts in bucket if ts >= cutoff]

            if len(bucket) >= self.max_requests:
                oldest = min(bucket)
                retry_after = max(1, int(self.window_seconds - (now - oldest)))
                self._buckets[key] = bucket
                return False, retry_after

            bucket.append(now)
            self._buckets[key] = bucket
            return True, 0


_DEFAULT_SECURITY_HEADERS: Final[dict[str, str]] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def _cors_origins_from_env() -> list[str]:
    raw = str(os.getenv("WEB_CORS_ORIGINS", "*") or "*").strip()
    if raw == "*":
        return ["*"]

    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


def _base_path_from_env() -> str:
    raw = str(os.getenv("WEB_BASE_PATH", "") or "").strip()
    if not raw or raw == "/":
        return ""

    if not raw.startswith("/"):
        raw = f"/{raw}"

    return raw.rstrip("/")


def _join_route(base_path: str, route: str) -> str:
    normalized_route = route if route.startswith("/") else f"/{route}"
    if not base_path:
        return normalized_route
    return f"{base_path}{normalized_route}"


def _frontend_dist_from_env() -> Path:
    explicit = str(os.getenv("WEB_STATIC_DIR", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    return (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()


def create_app() -> FastAPI:
    base_path = _base_path_from_env()
    api_prefix = _join_route(base_path, "/api")
    docs_path = _join_route(base_path, "/docs")
    redoc_path = _join_route(base_path, "/redoc")
    openapi_path = _join_route(base_path, "/openapi.json")

    frontend_dist = _frontend_dist_from_env()
    frontend_index = frontend_dist / "index.html"
    has_frontend = frontend_index.is_file()

    rate_limit_enabled = _env_bool("WEB_RATE_LIMIT_ENABLED", True)
    rate_limit_requests = _env_int("WEB_RATE_LIMIT_REQUESTS_PER_MINUTE", 30)
    rate_limit_window = _env_int("WEB_RATE_LIMIT_WINDOW_SECONDS", 60)
    rate_limiter = _FixedWindowRateLimiter(
        max_requests=rate_limit_requests,
        window_seconds=rate_limit_window,
    )

    enable_hsts = _env_bool("WEB_ENABLE_HSTS", False)

    app = FastAPI(
        title="GAIA Agent API",
        version="0.1.0",
        summary="FastAPI backend for GAIA LangGraph chat and trace inspection",
        docs_url=docs_path,
        redoc_url=redoc_path,
        openapi_url=openapi_path,
    )

    origins = _cors_origins_from_env()
    wildcard_origin = len(origins) == 1 and origins[0] == "*"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not wildcard_origin,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        for name, value in _DEFAULT_SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)

        if enable_hsts:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

        return response

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if not rate_limit_enabled:
            return await call_next(request)

        is_chat_call = request.method == "POST" and request.url.path in {
            _join_route(base_path, "/api/chat"),
            _join_route(base_path, "/api/chat/stream"),
        }

        if not is_chat_call:
            return await call_next(request)

        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "unknown"

        key = f"{client_ip}:{request.url.path}"
        allowed, retry_after = rate_limiter.allow(key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded for chat endpoint.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    app.include_router(health_router, prefix=api_prefix)
    app.include_router(chat_router, prefix=api_prefix)
    app.include_router(traces_router, prefix=api_prefix)

    @app.get("/", include_in_schema=False, response_model=None)
    def root():
        if has_frontend and not base_path:
            return FileResponse(frontend_index)

        return {
            "name": "gaia-agent-api",
            "base_path": base_path or "/",
            "docs": docs_path,
            "health": _join_route(base_path, "/api/health"),
            "web": _join_route(base_path, "/") if has_frontend else "",
        }

    @app.get(api_prefix)
    def api_root() -> dict[str, str]:
        return {
            "message": "GAIA Agent API is running",
            "base_path": base_path or "/",
            "health": _join_route(base_path, "/api/health"),
            "chat": _join_route(base_path, "/api/chat"),
            "chat_stream": _join_route(base_path, "/api/chat/stream"),
            "traces": _join_route(base_path, "/api/traces"),
        }

    if has_frontend:
        if base_path:

            @app.get(base_path, include_in_schema=False)
            @app.get(_join_route(base_path, "/"), include_in_schema=False)
            def prefixed_web_root() -> FileResponse:
                return FileResponse(frontend_index)

        catch_all_route = _join_route(base_path, "/{full_path:path}") if base_path else "/{full_path:path}"

        @app.get(catch_all_route, include_in_schema=False)
        def serve_frontend(full_path: str) -> FileResponse:
            if full_path.startswith("api/") or full_path in {"api", "docs", "redoc", "openapi.json"}:
                raise HTTPException(status_code=404, detail="Not found")

            candidate = (frontend_dist / full_path).resolve()
            try:
                candidate.relative_to(frontend_dist)
            except ValueError as err:
                raise HTTPException(status_code=404, detail="Not found") from err

            if candidate.is_file():
                return FileResponse(candidate)

            return FileResponse(frontend_index)

    return app


app = create_app()
