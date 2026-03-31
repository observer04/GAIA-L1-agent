from __future__ import annotations

import importlib
import os

modal = importlib.import_module("modal")

app = modal.App("gaia-agent-api")


def _runtime_env() -> dict[str, str]:
    env: dict[str, str] = {
        "WEB_BASE_PATH": os.getenv("WEB_BASE_PATH", "/gaia_agent"),
        "WEB_PUBLIC_MODE": os.getenv("WEB_PUBLIC_MODE", "true"),
        "AGENT_ALLOW_UNSAFE_TOOLS": os.getenv("AGENT_ALLOW_UNSAFE_TOOLS", "false"),
        "WEB_RATE_LIMIT_ENABLED": os.getenv("WEB_RATE_LIMIT_ENABLED", "true"),
        "WEB_RATE_LIMIT_REQUESTS_PER_MINUTE": os.getenv("WEB_RATE_LIMIT_REQUESTS_PER_MINUTE", "30"),
        "WEB_RATE_LIMIT_WINDOW_SECONDS": os.getenv("WEB_RATE_LIMIT_WINDOW_SECONDS", "60"),
        "WEB_ENABLE_HSTS": os.getenv("WEB_ENABLE_HSTS", "true"),
        "WEB_CORS_ORIGINS": os.getenv("WEB_CORS_ORIGINS", "*"),
    }

    passthrough_keys = [
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "TAVILY_API_KEY",
        "GOOGLE_SEARCH_API_KEY",
        "GOOGLE_CSE_ID",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ]

    for key in passthrough_keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            env[key] = value

    return env


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("agent", "backend")
)


@app.function(
    image=image,
    timeout=1800,
    secrets=[modal.Secret.from_dict(_runtime_env())],
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    from backend.main import create_app

    return create_app()


if __name__ == "__main__":
    print("Deploy with: modal deploy modal_app.py")