from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AgentConfig:
    api_url: str = "https://agents-course-unit4-scoring.hf.space"
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 1.0
    google_api_key: str = ""
    timeout_seconds: int = 240
    recursion_limit: int = 60
    max_iterations: int = 12
    level1_max_iterations: int = 12
    max_web_results_chars: int = 6000
    search_providers: str = "tavily,google,ddg"
    search_provider_timeout_seconds: int = 8
    tavily_api_key: str = ""
    tavily_search_depth: str = "basic"
    tavily_include_answer: bool = True
    tavily_include_raw_content: bool = False
    local_working_dir: str = "/tmp/gaia_agent"
    enable_langsmith_tracing: bool = False
    langsmith_project: str = "gaia-langgraph-local"

    @staticmethod
    def _as_bool(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "AgentConfig":
        working_dir = os.getenv("GAIA_WORKING_DIR", cls.local_working_dir)
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        tracing_enabled = cls._as_bool(os.getenv("LANGSMITH_TRACING"), False) or cls._as_bool(
            os.getenv("LANGCHAIN_TRACING_V2"), False
        )

        return cls(
            api_url=os.getenv("GAIA_API_URL", cls.api_url),
            model_name=os.getenv("GEMINI_MODEL", cls.model_name),
            temperature=_get_float("GEMINI_TEMPERATURE", cls.temperature),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            timeout_seconds=_get_int("AGENT_TIMEOUT_SECONDS", cls.timeout_seconds),
            recursion_limit=_get_int("AGENT_RECURSION_LIMIT", cls.recursion_limit),
            max_iterations=_get_int("AGENT_MAX_ITERATIONS", cls.max_iterations),
            level1_max_iterations=_get_int("AGENT_LEVEL1_MAX_ITERATIONS", cls.level1_max_iterations),
            max_web_results_chars=_get_int("MAX_WEB_RESULTS_CHARS", cls.max_web_results_chars),
            search_providers=os.getenv("SEARCH_PROVIDERS", cls.search_providers),
            search_provider_timeout_seconds=_get_int(
                "SEARCH_PROVIDER_TIMEOUT_SECONDS",
                cls.search_provider_timeout_seconds,
            ),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            tavily_search_depth=os.getenv("TAVILY_SEARCH_DEPTH", cls.tavily_search_depth),
            tavily_include_answer=cls._as_bool(
                os.getenv("TAVILY_INCLUDE_ANSWER"),
                cls.tavily_include_answer,
            ),
            tavily_include_raw_content=cls._as_bool(
                os.getenv("TAVILY_INCLUDE_RAW_CONTENT"),
                cls.tavily_include_raw_content,
            ),
            local_working_dir=working_dir,
            enable_langsmith_tracing=tracing_enabled,
            langsmith_project=os.getenv("LANGSMITH_PROJECT", cls.langsmith_project),
        )

    def assert_required(self) -> None:
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for Gemini model access. "
                "Set it in your environment or .env file."
            )
