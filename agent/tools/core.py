from __future__ import annotations

import concurrent.futures
import base64
import os
import re
import html
import tempfile
import importlib
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, quote, unquote, urlencode, urlparse
from typing import Any, Dict, List

import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import AgentConfig
from .python_executor import BashCommandExecutor, StatefulPythonExecutor
from .runtime import get_runtime_task_id, get_runtime_working_dir

DEFAULT_API_URL = AgentConfig.from_env().api_url

_GAIA_DATASET_ID = "gaia-benchmark/GAIA"
_GAIA_DATASET_RESOLVE_MAIN_URL = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main"
_GAIA_DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_GAIA_DATASET_DEFAULT_CONFIG = "2023_all"
_GAIA_DATASET_DEFAULT_SPLIT = "validation"
_GAIA_DATASET_ROWS_PAGE_SIZE = 100
_GAIA_ATTACHMENT_COMMON_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".pdf",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".zip",
    ".docx",
    ".pptx",
    ".json",
    ".jsonld",
    ".txt",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".pdb",
)
_HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
_GAIA_ATTACHMENT_INDEX_CACHE: dict[str, dict[str, dict[str, str]]] = {}

_python_executor = StatefulPythonExecutor()
_bash_executor = BashCommandExecutor(timeout_seconds=15)
_VLM_CLIENTS: dict[str, ChatGoogleGenerativeAI] = {}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_LEAK_URL_TOKENS = (
    "gaia_20.jsonl",
    "test_questions.json",
    "questions.json",
    "questions_data.txt",
    "metadata.jsonl",
)
_LOW_QUALITY_SOURCE_DOMAINS = (
    "proprofs.com",
    "studypool.com",
    "coursehero.com",
    "chegg.com",
    "quizlet.com",
    "brainly.com",
    "reddit.com",
    "quora.com",
)
_TRACKING_QUERY_PARAM_PREFIXES = (
    "utm_",
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
)
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
_AUDIO_TRANSCRIBER: Any | None = None
_SEARCH_PROVIDER_ALIASES = {
    "ddg": "ddg",
    "duckduckgo": "ddg",
    "duckduckgo-search": "ddg",
    "google": "google",
    "google-genai": "google",
    "google_genai": "google",
    "google_cse": "google",
    "tavily": "tavily",
}
_SEARCH_PROVIDER_WEIGHTS = {
    "google": 3,
    "tavily": 3,
    "ddg": 2,
}


def _safe_filename(raw_name: str, fallback: str) -> str:
    name = (raw_name or "").strip().strip("\"'")
    if not name:
        return fallback
    return os.path.basename(name)


def _truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


def _truncate_wikipedia_extract(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw

    lowered = raw.lower()
    focus_markers = [
        "== discography ==",
        "== studio albums ==",
        "== filmography ==",
        "== bibliography ==",
    ]

    intro_budget = max(1500, int(max_chars * 0.45))
    intro = raw[:intro_budget]

    focus_chunk = ""
    remaining = max_chars - len(intro)
    if remaining > 600:
        for marker in focus_markers:
            idx = lowered.find(marker)
            if idx == -1:
                continue
            chunk_budget = min(remaining - 40, max(800, int(max_chars * 0.5)))
            start = max(0, idx - 120)
            end = min(len(raw), start + chunk_budget)
            focus_chunk = "\n\n[Focused section]\n" + raw[start:end]
            break

    combined = intro + focus_chunk
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined + "\n...[TRUNCATED]"


def _resolve_path(file_path: str) -> Path:
    candidate = Path((file_path or "").strip())
    if candidate.is_absolute():
        return candidate

    runtime_dir = get_runtime_working_dir().strip()
    if runtime_dir:
        resolved = Path(runtime_dir) / candidate
        if resolved.exists():
            return resolved
    return candidate


def _default_download_dir() -> Path:
    runtime_dir = get_runtime_working_dir().strip()
    if runtime_dir:
        path = Path(runtime_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    path = Path(tempfile.gettempdir()) / "gaia_task_files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_youtube_video_id(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower().replace("www.", "")
    if host in {"youtube.com", "m.youtube.com"}:
        query = parse_qs(parsed.query)
        return (query.get("v", [""])[0] or "").strip()
    if host == "youtu.be":
        return parsed.path.lstrip("/").strip()
    return ""


def _extract_gaia_task_id(text: str) -> str:
    match = _UUID_PATTERN.search(str(text or ""))
    return match.group(0) if match else ""


def _extract_wikipedia_title_from_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if "wikipedia.org" not in host or "/wiki/" not in parsed.path:
        return ""

    slug = parsed.path.split("/wiki/", 1)[1].strip()
    if not slug:
        return ""
    return unquote(slug)


def _strip_xml_text(xml_blob: str) -> str:
    text = str(xml_blob or "")
    if not text:
        return ""

    text = re.sub(r"</(?:w:p|a:p|text:p|p)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _duckduckgo_instant_fallback(query: str, max_items: int, max_chars: int) -> str:
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
    }

    response = requests.get(
        "https://api.duckduckgo.com/",
        params=params,
        timeout=25,
        headers=REQUEST_HEADERS,
    )
    response.raise_for_status()
    payload_raw = response.json()
    payload = payload_raw if isinstance(payload_raw, dict) else {}

    lines: list[str] = []

    heading = str(payload.get("Heading", "")).strip()
    abstract_text = str(payload.get("AbstractText", "")).strip()
    abstract_url = str(payload.get("AbstractURL", "")).strip() or "(no url)"
    if abstract_text and not _is_potential_benchmark_leak_url(abstract_url):
        title = heading or "DuckDuckGo Instant Answer"
        lines.append(f"1. Title: {title}\nURL: {abstract_url}\nSnippet: {abstract_text}")

    next_index = len(lines) + 1

    def append_topic(topic: dict[str, Any]) -> None:
        nonlocal next_index
        if next_index > max_items:
            return

        text = str(topic.get("Text", "")).strip()
        if not text:
            return

        first_url = str(topic.get("FirstURL", "")).strip() or "(no url)"
        if _is_potential_benchmark_leak_url(first_url):
            return

        title = text.split(" - ", 1)[0].strip() or "(no title)"
        lines.append(f"{next_index}. Title: {title}\nURL: {first_url}\nSnippet: {text}")
        next_index += 1

    related_topics = payload.get("RelatedTopics")
    if isinstance(related_topics, list):
        for item in related_topics:
            if next_index > max_items:
                break
            if not isinstance(item, dict):
                continue

            nested_topics = item.get("Topics")
            if isinstance(nested_topics, list):
                for nested in nested_topics:
                    if isinstance(nested, dict):
                        append_topic(nested)
                    if next_index > max_items:
                        break
            else:
                append_topic(item)

    if not lines:
        return "No search results found."
    return _truncate("\n\n".join(lines), max_chars=max_chars)


def _is_potential_benchmark_leak_url(url: str) -> bool:
    lowered = str(url or "").strip().lower()
    if not lowered:
        return False

    if "agents-course-unit4-scoring.hf.space/questions" in lowered:
        return True

    if any(token in lowered for token in _LEAK_URL_TOKENS):
        return True

    if "huggingface.co/datasets/gaia-benchmark/gaia" in lowered:
        return True
    if "hf.co/datasets/gaia-benchmark/gaia" in lowered:
        return True

    if "huggingface.co/datasets/" in lowered and any(
        marker in lowered
        for marker in ["gaia", "benchmark", "sft", "agent", "final-assignment", "evaluation", "eval"]
    ):
        return True

    if "gaia" in lowered and (
        "huggingface.co/spaces/" in lowered
        or "raw.githubusercontent.com/" in lowered
        or "github.com/" in lowered
    ):
        if any(marker in lowered for marker in ["questions", "answer", ".json", ".jsonl", ".txt"]):
            return True

    if (
        "huggingface.co/spaces/" in lowered
        or "raw.githubusercontent.com/" in lowered
        or "github.com/" in lowered
    ) and any(marker in lowered for marker in ["final-assignment", "agents-course-gaia", "gaia-benchmark"]):
        return True

    if (
        "raw.githubusercontent.com/" in lowered
        or "github.com/" in lowered
        or "huggingface.co/spaces/" in lowered
    ) and re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.", lowered):
        return True

    if _UUID_PATTERN.search(lowered) and (
        "github.com/" in lowered
        or "raw.githubusercontent.com/" in lowered
        or "huggingface.co/spaces/" in lowered
        or "hf.co/spaces/" in lowered
    ):
        return True

    if (
        "huggingface.co/spaces/" in lowered
        and "gaia" in lowered
        and any(marker in lowered for marker in ["/commit/", "/blob/", "/raw/"])
    ):
        return True

    if (
        "huggingface.co/spaces/" in lowered
        and any(marker in lowered for marker in ["/commit/", "/blob/", "/raw/"])
        and any(
            marker in lowered
            for marker in ["final_assignment_template", "final-assignment-template", "agents-course", "unit4"]
        )
    ):
        return True

    return False


def _contains_benchmark_leak_url_in_text(text: str) -> bool:
    for match in re.finditer(r"https?://[^\s)\]>'\"]+", str(text or ""), flags=re.IGNORECASE):
        candidate = match.group(0).rstrip(".,;:")
        if _is_potential_benchmark_leak_url(candidate):
            return True
    return False


def _is_potential_benchmark_leak_content(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False

    has_task_id = lowered.count('"task_id"') >= 1
    has_answer_fields = any(
        marker in lowered
        for marker in ['"submitted_answer"', '"final answer"', '"final_answer"', '"answer"']
    )
    if has_task_id and has_answer_fields:
        return True

    has_task_ids = lowered.count('"task_id"') >= 2
    has_final_answer_keys = lowered.count('"final answer"') >= 2 or lowered.count('"final_answer"') >= 2
    if has_task_ids and has_final_answer_keys:
        return True

    if (
        "gaia-benchmark/gaia" in lowered
        and "discussion" in lowered
        and ("final answer" in lowered or "task_id" in lowered)
    ):
        return True

    return False


def _is_low_quality_source_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    domain = parsed.netloc.lower().replace("www.", "")
    if not domain:
        return False

    if domain in _LOW_QUALITY_SOURCE_DOMAINS:
        return True
    return any(domain.endswith(f".{suffix}") for suffix in _LOW_QUALITY_SOURCE_DOMAINS)


def _resolve_search_providers(configured: str) -> list[str]:
    providers: list[str] = []
    for token in str(configured or "").split(","):
        normalized = _SEARCH_PROVIDER_ALIASES.get(token.strip().lower())
        if not normalized:
            continue
        if normalized not in providers:
            providers.append(normalized)

    if not providers:
        return ["ddg"]
    return providers


def _search_hit(
    title: str,
    url: str,
    snippet: str,
    provider: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    safe_title = re.sub(r"\s+", " ", str(title or "").strip()) or "(no title)"
    safe_url = str(url or "").strip() or "(no url)"
    safe_snippet = re.sub(r"\s+", " ", str(snippet or "").strip()) or "(no snippet)"
    hit: dict[str, Any] = {
        "title": safe_title,
        "url": safe_url,
        "snippet": safe_snippet,
        "provider": str(provider or "unknown").strip() or "unknown",
    }
    if confidence is not None:
        try:
            hit["confidence"] = float(confidence)
        except (TypeError, ValueError):
            pass
    return hit


def _first_url_from_text(text: str) -> str:
    match = re.search(r"https?://[^\s)\]>'\"]+", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(0).rstrip(".,;:")


def _url_dedupe_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw or raw == "(no url)":
        return raw.lower()

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw.lower().rstrip("/")

    path = parsed.path.rstrip("/")
    key = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"

    normalized_query_pairs = []
    for param_key, param_value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = str(param_key or "").strip().lower()
        if not normalized_key:
            continue
        if any(normalized_key.startswith(prefix) for prefix in _TRACKING_QUERY_PARAM_PREFIXES):
            continue
        normalized_query_pairs.append((normalized_key, str(param_value or "").strip()))

    if normalized_query_pairs:
        normalized_query_pairs.sort()
        key = f"{key}?{urlencode(normalized_query_pairs, doseq=True)}"

    return key


def _provider_weight(provider: str) -> int:
    return int(_SEARCH_PROVIDER_WEIGHTS.get(str(provider or "").strip().lower(), 1))


def _search_ddg_hits(query: str, max_items: int, timeout_seconds: int = 8) -> list[dict[str, str]]:
    del timeout_seconds
    errors: list[str] = []

    try:
        ddgs_module = importlib.import_module("ddgs")
        DDGS = getattr(ddgs_module, "DDGS")

        with DDGS() as search_client:
            raw_items = list(search_client.text(query, max_results=max_items))

        hits: list[dict[str, str]] = []
        for item in raw_items:
            title = str(item.get("title") or item.get("heading") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if not url or _is_potential_benchmark_leak_url(url):
                continue
            hits.append(_search_hit(title, url, snippet, provider="ddg"))

        if hits:
            return hits[:max_items]
    except Exception as err:  # noqa: BLE001
        errors.append(f"ddgs: {err}")

    try:
        ddg_search = DuckDuckGoSearchRun()
        raw_result = str(ddg_search.run(query)).strip()
        if raw_result and not _contains_benchmark_leak_url_in_text(raw_result):
            first_url = _first_url_from_text(raw_result)
            if first_url and _is_potential_benchmark_leak_url(first_url):
                first_url = ""
            return [
                _search_hit(
                    title="DuckDuckGo Search Result",
                    url=first_url,
                    snippet=raw_result,
                    provider="ddg",
                )
            ]
    except Exception as err:  # noqa: BLE001
        errors.append(f"langchain_ddg: {err}")

    try:
        fallback_text = _duckduckgo_instant_fallback(
            query=query,
            max_items=max_items,
            max_chars=4000,
        )
        if fallback_text and fallback_text != "No search results found." and not fallback_text.startswith("ERROR:"):
            return [
                _search_hit(
                    title="DuckDuckGo Instant Fallback",
                    url="",
                    snippet=fallback_text,
                    provider="ddg",
                )
            ]
    except Exception as err:  # noqa: BLE001
        errors.append(f"instant_fallback: {err}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return []


def _search_google_genai_hits(
    query: str,
    max_items: int,
    timeout_seconds: int = 8,
) -> list[dict[str, str]]:
    del timeout_seconds

    api_key = AgentConfig.from_env().google_api_key or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        genai_module = importlib.import_module("google.genai")
        types_module = importlib.import_module("google.genai.types")
        Client = getattr(genai_module, "Client")
        Tool = getattr(types_module, "Tool")
        GoogleSearch = getattr(types_module, "GoogleSearch")
        GoogleSearchRetrieval = getattr(types_module, "GoogleSearchRetrieval", None)
        GenerateContentConfig = getattr(types_module, "GenerateContentConfig")
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"google_genai_import: {err}") from err

    config = AgentConfig.from_env()
    model_name = str(os.getenv("GEMINI_SEARCH_MODEL", config.model_name)).strip() or config.model_name
    search_prompt = (
        "Search the web for the query below and provide concise, source-grounded findings.\n"
        f"Query: {query}"
    )

    tool_variants: list[Any] = []
    tool_variants.append(Tool(google_search=GoogleSearch()))
    if GoogleSearchRetrieval is not None:
        try:
            tool_variants.append(Tool(google_search_retrieval=GoogleSearchRetrieval()))
        except Exception:
            pass

    errors: list[str] = []
    for variant in tool_variants:
        variant_name = "google_search_retrieval" if getattr(variant, "google_search_retrieval", None) else "google_search"
        try:
            client = Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=search_prompt,
                config=GenerateContentConfig(tools=[variant], temperature=0.0),
            )
        except Exception as err:  # noqa: BLE001
            errors.append(f"{variant_name}: {err}")
            continue

        response_text = str(getattr(response, "text", "") or "").strip()
        snippet = _truncate(response_text or f"Google search evidence for query: {query}", max_chars=1100)

        hits: list[dict[str, str]] = []
        seen_keys: set[str] = set()

        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            grounding_chunks = getattr(grounding_metadata, "grounding_chunks", []) if grounding_metadata else []
            for chunk in grounding_chunks or []:
                web_chunk = getattr(chunk, "web", None)
                if web_chunk is None:
                    continue

                url = str(getattr(web_chunk, "uri", "") or "").strip()
                title = str(getattr(web_chunk, "title", "") or "").strip() or "(no title)"
                if not url or _is_potential_benchmark_leak_url(url):
                    continue

                key = _url_dedupe_key(url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                hits.append(_search_hit(title, url, snippet, provider="google"))
                if len(hits) >= max_items:
                    break
            if len(hits) >= max_items:
                break

        if hits:
            return hits[:max_items]

        if response_text:
            for match in re.finditer(r"https?://[^\s)\]>'\"]+", response_text, flags=re.IGNORECASE):
                url = match.group(0).rstrip(".,;:")
                if not url or _is_potential_benchmark_leak_url(url):
                    continue
                key = _url_dedupe_key(url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                hits.append(
                    _search_hit(
                        title="Google Search grounded source",
                        url=url,
                        snippet=snippet,
                        provider="google",
                    )
                )
                if len(hits) >= max_items:
                    break

        if hits:
            return hits[:max_items]

    if errors:
        raise RuntimeError("; ".join(errors))
    return []


def _search_google_cse_hits(
    query: str,
    max_items: int,
    timeout_seconds: int = 8,
) -> list[dict[str, str]]:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()
    if not api_key or not cse_id:
        return []

    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": max(1, min(int(max_items), 10)),
        },
        timeout=max(2, int(timeout_seconds)),
        headers=REQUEST_HEADERS,
    )
    response.raise_for_status()

    payload_raw = response.json()
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []

    hits: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("link", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if not url or _is_potential_benchmark_leak_url(url):
            continue
        hits.append(_search_hit(title, url, snippet, provider="google"))

    return hits[:max_items]


def _search_google_custom_search_hits(
    query: str,
    max_items: int,
    timeout_seconds: int = 8,
) -> list[dict[str, str]]:
    errors: list[str] = []

    try:
        genai_hits = _search_google_genai_hits(
            query=query,
            max_items=max_items,
            timeout_seconds=timeout_seconds,
        )
        if genai_hits:
            return genai_hits
    except Exception as err:  # noqa: BLE001
        errors.append(f"google_genai: {err}")

    try:
        cse_hits = _search_google_cse_hits(
            query=query,
            max_items=max_items,
            timeout_seconds=timeout_seconds,
        )
        if cse_hits:
            return cse_hits
    except Exception as err:  # noqa: BLE001
        errors.append(f"google_cse: {err}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return []


def _search_tavily_hits(query: str, max_items: int, timeout_seconds: int = 8) -> list[dict[str, Any]]:
    config = AgentConfig.from_env()
    api_key = str(config.tavily_api_key or os.getenv("TAVILY_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("missing TAVILY_API_KEY")

    search_depth = str(config.tavily_search_depth or "basic").strip().lower()
    if search_depth not in {"basic", "advanced"}:
        search_depth = "basic"

    include_answer = bool(config.tavily_include_answer)
    include_raw_content = bool(config.tavily_include_raw_content)

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max(1, min(int(max_items), 10)),
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_images": False,
            "include_raw_content": include_raw_content,
        },
        timeout=max(2, int(timeout_seconds)),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": REQUEST_HEADERS["User-Agent"],
        },
    )
    response.raise_for_status()

    payload_raw = response.json()
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    tavily_answer = str(payload.get("answer", "") or "").strip()
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []

    hits: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        snippet = str(item.get("content", "") or "").strip()
        if include_raw_content:
            raw_content = str(item.get("raw_content", "") or "").strip()
            if raw_content:
                raw_content = _truncate(raw_content, max_chars=1000)
                snippet = f"{snippet}\nRaw content excerpt: {raw_content}" if snippet else raw_content

        score_raw = item.get("score")
        score: float | None = None
        try:
            if score_raw is not None:
                score = float(score_raw)
        except (TypeError, ValueError):
            score = None

        if not url or _is_potential_benchmark_leak_url(url):
            continue

        hits.append(_search_hit(title, url, snippet, provider="tavily", confidence=score))

    if tavily_answer and not _is_potential_benchmark_leak_content(tavily_answer):
        if hits:
            first_snippet = str(hits[0].get("snippet", "") or "")
            hits[0]["snippet"] = _truncate(
                f"Tavily answer summary: {tavily_answer}\n\n{first_snippet}",
                max_chars=1400,
            )
        else:
            hits.append(
                _search_hit(
                    title="Tavily answer summary",
                    url="",
                    snippet=tavily_answer,
                    provider="tavily",
                )
            )

    return hits[:max_items]


def _run_parallel_provider_search(
    query: str,
    max_items: int,
    providers: list[str],
    timeout_seconds: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    handlers = {
        "ddg": _search_ddg_hits,
        "google": _search_google_custom_search_hits,
        "tavily": _search_tavily_hits,
    }

    hits_by_provider: dict[str, list[dict[str, Any]]] = {}
    provider_errors: dict[str, str] = {}

    max_workers = max(1, len(providers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[concurrent.futures.Future[list[dict[str, Any]]], str] = {}
        for provider in providers:
            handler = handlers.get(provider)
            if handler is None:
                provider_errors[provider] = "unsupported provider"
                continue
            futures[executor.submit(handler, query, max_items, timeout_seconds)] = provider

        for future in concurrent.futures.as_completed(futures):
            provider = futures[future]
            try:
                provider_hits = future.result()
            except Exception as err:  # noqa: BLE001
                provider_errors[provider] = f"{err.__class__.__name__}: {err}"
                continue

            if provider_hits:
                hits_by_provider[provider] = provider_hits

    return hits_by_provider, provider_errors


def _merge_and_rank_search_hits(
    hits_by_provider: dict[str, list[dict[str, Any]]],
    max_items: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for provider, hits in hits_by_provider.items():
        for hit in hits:
            title = str(hit.get("title", "")).strip() or "(no title)"
            url = str(hit.get("url", "")).strip() or "(no url)"
            snippet = str(hit.get("snippet", "")).strip() or "(no snippet)"

            if url != "(no url)" and _is_potential_benchmark_leak_url(url):
                continue
            if _contains_benchmark_leak_url_in_text(snippet) or _is_potential_benchmark_leak_content(snippet):
                continue

            key = _url_dedupe_key(url)
            if not key:
                continue

            existing = merged.get(key)
            if existing is None:
                existing = {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "providers": set(),
                    "score": 0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0,
                }
                merged[key] = existing

            providers_set = existing["providers"]
            if isinstance(providers_set, set):
                providers_set.add(provider)

            existing["score"] = float(existing.get("score", 0.0)) + float(_provider_weight(provider))

            confidence_raw = hit.get("confidence")
            if confidence_raw is None:
                confidence = None
            else:
                try:
                    confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence = None
            if confidence is not None:
                bounded_confidence = max(0.0, min(1.0, confidence))
                existing["confidence_sum"] = float(existing.get("confidence_sum", 0.0)) + bounded_confidence
                existing["confidence_count"] = int(existing.get("confidence_count", 0)) + 1
                existing["score"] += bounded_confidence * 2.0

            if url != "(no url)" and _is_low_quality_source_url(url):
                existing["score"] -= 2
            else:
                existing["score"] += 1

            if len(snippet) > len(str(existing.get("snippet", ""))):
                existing["snippet"] = snippet
            if title and title != "(no title)" and str(existing.get("title", "")) == "(no title)":
                existing["title"] = title

    ranked: list[dict[str, Any]] = []
    for item in merged.values():
        providers_set = item.get("providers", set())
        providers = sorted(str(p).strip() for p in providers_set if str(p).strip())
        item["providers"] = providers
        item["score"] = float(item.get("score", 0.0)) + float(max(0, len(providers) - 1))

        confidence_count = int(item.get("confidence_count", 0))
        confidence_sum = float(item.get("confidence_sum", 0.0))
        if confidence_count > 0:
            item["confidence"] = round(confidence_sum / confidence_count, 4)

        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            len(item.get("providers", [])),
            len(str(item.get("snippet", ""))),
        ),
        reverse=True,
    )

    return ranked[: max(1, int(max_items))]


def _format_search_hits_for_output(hits: list[dict[str, Any]], max_chars: int) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        providers = hit.get("providers", [])
        if isinstance(providers, list):
            provider_text = ", ".join(str(p) for p in providers if str(p).strip())
        else:
            provider_text = str(providers)
        provider_text = provider_text or "unknown"

        lines.append(
            f"{idx}. Title: {hit.get('title', '(no title)')}\n"
            f"URL: {hit.get('url', '(no url)')}\n"
            f"Provider(s): {provider_text}\n"
            f"Snippet: {hit.get('snippet', '(no snippet)')}"
        )

    if not lines:
        return "No search results found."
    return _truncate("\n\n".join(lines), max_chars=max_chars)


def _format_provider_warnings(provider_errors: dict[str, str]) -> str:
    if not provider_errors:
        return ""

    lines = ["Provider warnings:"]
    for provider_name, message in sorted(provider_errors.items()):
        lines.append(f"- {provider_name}: {message}")
    return "\n".join(lines)


def _get_audio_transcriber() -> Any:
    global _AUDIO_TRANSCRIBER
    if _AUDIO_TRANSCRIBER is not None:
        return _AUDIO_TRANSCRIBER

    transformers_module = importlib.import_module("transformers")
    pipeline = getattr(transformers_module, "pipeline")
    model_name = os.getenv("GAIA_ASR_MODEL", "openai/whisper-tiny.en")
    _AUDIO_TRANSCRIBER = pipeline("automatic-speech-recognition", model=model_name)
    return _AUDIO_TRANSCRIBER


def _guess_image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    chunks.append(str(text_value))
        return "\n".join(chunks).strip()
    return str(content or "")


def _get_vlm_client(model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
    model = (model_name or "").strip() or "gemini-2.5-flash"
    cache_key = f"{model}|{bool(api_key)}"
    client = _VLM_CLIENTS.get(cache_key)
    if client is not None:
        return client

    client = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        google_api_key=api_key,
        timeout=min(90, AgentConfig.from_env().timeout_seconds),
    )
    _VLM_CLIENTS[cache_key] = client
    return client


def _fetch_wikipedia_extract(url: str, max_chars: int) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if "wikipedia.org" not in host or "/wiki/" not in parsed.path:
        return ""

    decoded_title = _extract_wikipedia_title_from_url(url)
    if not decoded_title:
        return ""

    if decoded_title.startswith("Template:"):
        raw_url = (
            f"{parsed.scheme or 'https'}://{parsed.netloc}/w/index.php"
            f"?title={quote(decoded_title)}&action=raw"
        )
        try:
            raw_response = requests.get(raw_url, timeout=30, headers=REQUEST_HEADERS)
            raw_response.raise_for_status()
            raw_text = raw_response.text.strip()
            if raw_text:
                return _truncate(raw_text, max_chars=max_chars)
        except Exception:  # noqa: BLE001
            pass

    api_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "titles": decoded_title,
    }

    try:
        response = requests.get(api_url, params=params, timeout=30, headers=REQUEST_HEADERS)
        response.raise_for_status()
        payload = response.json()
        pages = (payload.get("query", {}) or {}).get("pages", {}) or {}
        for page_data in pages.values():
            title = str(page_data.get("title", "")).strip()
            extract = str(page_data.get("extract", "")).strip()
            if extract:
                focused_extract = _truncate_wikipedia_extract(extract, max_chars=max_chars)
                if title:
                    return _truncate(f"Title: {title}\n\n{focused_extract}", max_chars=max_chars)
                return focused_extract
    except Exception:  # noqa: BLE001
        return ""

    return ""


@tool
def fetch_wikipedia_section(url: str, section_name: str = "Discography", max_chars: int = 12000) -> str:
    """Fetch a specific section from a Wikipedia page (e.g., Discography, Filmography)."""
    cleaned_url = (url or "").strip()
    section_label = (section_name or "").strip() or "Discography"

    title = _extract_wikipedia_title_from_url(cleaned_url)
    if not title:
        return "ERROR: URL is not a valid Wikipedia article URL"

    parsed = urlparse(cleaned_url)
    api_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/w/api.php"

    try:
        sections_resp = requests.get(
            api_url,
            params={
                "action": "parse",
                "page": title,
                "prop": "sections",
                "format": "json",
            },
            timeout=30,
            headers=REQUEST_HEADERS,
        )
        sections_resp.raise_for_status()
        sections_payload = sections_resp.json() if isinstance(sections_resp.json(), dict) else {}
        sections = (sections_payload.get("parse", {}) or {}).get("sections", [])
        if not isinstance(sections, list) or not sections:
            return f"ERROR: no sections found for page ({title})"

        target_section = None
        target_lower = section_label.lower()
        for section in sections:
            line = str(section.get("line", "")).strip()
            if line.lower() == target_lower:
                target_section = section
                break

        if target_section is None:
            for section in sections:
                line = str(section.get("line", "")).strip()
                if target_lower in line.lower() or line.lower() in target_lower:
                    target_section = section
                    break

        if target_section is None:
            available = ", ".join(str(s.get("line", "")).strip() for s in sections[:10] if str(s.get("line", "")).strip())
            return (
                f"ERROR: section '{section_label}' not found on page '{title}'. "
                f"Available sections include: {available or 'none'}"
            )

        section_index = str(target_section.get("index", "")).strip()
        if not section_index:
            return f"ERROR: unable to resolve section index for '{section_label}'"

        section_resp = requests.get(
            api_url,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "section": section_index,
                "format": "json",
            },
            timeout=30,
            headers=REQUEST_HEADERS,
        )
        section_resp.raise_for_status()
        section_payload = section_resp.json() if isinstance(section_resp.json(), dict) else {}
        html_content = str(((section_payload.get("parse", {}) or {}).get("text", {}) or {}).get("*", ""))
        if not html_content:
            return f"ERROR: empty content for section '{section_label}'"

        try:
            bs4_module = importlib.import_module("bs4")
            BeautifulSoup = getattr(bs4_module, "BeautifulSoup")
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "noscript", "sup"]):
                tag.extract()
            text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html_content)
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return f"ERROR: no readable text in section '{section_label}'"

        rendered = f"Page: {title}\nSection: {section_label}\n\n{text}"
        return _truncate(rendered, max_chars=max_chars)
    except requests.RequestException as err:
        return f"ERROR: failed fetching Wikipedia section ({err})"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: unexpected Wikipedia section error ({err})"


def _fetch_with_jina_mirror(url: str, max_chars: int) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""

    mirror_url = f"https://r.jina.ai/{cleaned}"
    try:
        response = requests.get(mirror_url, timeout=45, headers=REQUEST_HEADERS)
        response.raise_for_status()
        return _truncate(response.text, max_chars=max_chars)
    except Exception:  # noqa: BLE001
        return ""


def _hf_token_from_env() -> str:
    for env_key in _HF_TOKEN_ENV_KEYS:
        token = str(os.getenv(env_key, "") or "").strip()
        if token:
            return token
    return ""


def _hf_dataset_auth_headers() -> dict[str, str]:
    token = _hf_token_from_env()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _gaia_dataset_configs() -> list[str]:
    raw_value = str(os.getenv("GAIA_DATASET_CONFIGS", _GAIA_DATASET_DEFAULT_CONFIG) or "").strip()
    configs: list[str] = []
    for item in raw_value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        if normalized not in configs:
            configs.append(normalized)

    if not configs:
        configs.append(_GAIA_DATASET_DEFAULT_CONFIG)
    return configs


def _gaia_dataset_split() -> str:
    split = str(os.getenv("GAIA_DATASET_SPLIT", _GAIA_DATASET_DEFAULT_SPLIT) or "").strip()
    return split or _GAIA_DATASET_DEFAULT_SPLIT


def _normalize_gaia_relative_path(path_value: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""

    if raw.lower().startswith(("http://", "https://")):
        if "/resolve/main/" in raw:
            raw = raw.split("/resolve/main/", 1)[1]
        else:
            parsed = urlparse(raw)
            raw = parsed.path

    raw = raw.lstrip("/")

    for prefix in (
        "datasets/gaia-benchmark/GAIA/",
        "gaia-benchmark/GAIA/",
        "resolve/main/",
        "blob/main/",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]

    return raw.strip()


def _gaia_attachment_cache_key(auth_headers: dict[str, str], configs: list[str], split: str) -> str:
    auth_mode = "auth" if auth_headers.get("Authorization") else "anon"
    return f"{auth_mode}|{','.join(configs)}|{split}"


def _build_gaia_attachment_index(auth_headers: dict[str, str]) -> tuple[dict[str, dict[str, str]], str]:
    configs = _gaia_dataset_configs()
    split = _gaia_dataset_split()
    cache_key = _gaia_attachment_cache_key(auth_headers, configs, split)

    cached = _GAIA_ATTACHMENT_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached, ""

    request_headers = dict(REQUEST_HEADERS)
    request_headers.update(auth_headers)

    attachment_index: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for config_name in configs:
        offset = 0
        while True:
            params = {
                "dataset": _GAIA_DATASET_ID,
                "config": config_name,
                "split": split,
                "offset": offset,
                "length": _GAIA_DATASET_ROWS_PAGE_SIZE,
            }

            try:
                response = requests.get(
                    _GAIA_DATASET_ROWS_URL,
                    params=params,
                    timeout=30,
                    headers=request_headers,
                )
            except requests.RequestException as err:
                errors.append(f"{config_name}: {err}")
                break

            if response.status_code in {401, 403}:
                errors.append(
                    f"{config_name}: dataset access denied (status {response.status_code})"
                )
                break

            try:
                response.raise_for_status()
            except requests.RequestException as err:
                errors.append(f"{config_name}: {err}")
                break

            payload_raw = response.json()
            payload = payload_raw if isinstance(payload_raw, dict) else {}
            rows = payload.get("rows", [])
            if not isinstance(rows, list) or not rows:
                break

            for row_item in rows:
                if not isinstance(row_item, dict):
                    continue
                row_data = row_item.get("row")
                if not isinstance(row_data, dict):
                    continue

                row_task_id = str(row_data.get("task_id", "") or "").strip()
                if not row_task_id:
                    continue

                row_file_name = _safe_filename(str(row_data.get("file_name", "") or ""), "")
                row_file_path = _normalize_gaia_relative_path(str(row_data.get("file_path", "") or ""))

                if not row_file_path and row_file_name:
                    row_file_path = _normalize_gaia_relative_path(f"2023/validation/{row_file_name}")

                if not row_file_name and not row_file_path:
                    continue

                attachment_index[row_task_id] = {
                    "file_name": row_file_name,
                    "file_path": row_file_path,
                    "config": config_name,
                    "split": split,
                }

            total_rows_raw = payload.get("num_rows_total")
            if isinstance(total_rows_raw, int):
                total_rows = total_rows_raw
            else:
                total_rows_text = str(total_rows_raw).strip() if total_rows_raw is not None else ""
                total_rows = int(total_rows_text) if total_rows_text.isdigit() else 0

            offset += len(rows)
            if total_rows and offset >= total_rows:
                break
            if len(rows) < _GAIA_DATASET_ROWS_PAGE_SIZE:
                break

    _GAIA_ATTACHMENT_INDEX_CACHE[cache_key] = attachment_index
    return attachment_index, "; ".join(errors)


def _gaia_attachment_entry_for_task(task_id: str, auth_headers: dict[str, str]) -> tuple[dict[str, str], str]:
    index, load_errors = _build_gaia_attachment_index(auth_headers=auth_headers)
    return dict(index.get(task_id, {})), load_errors


def _gaia_candidate_attachment_paths(
    task_id: str,
    expected_filename: str,
    entry: dict[str, str],
) -> list[str]:
    candidates: list[str] = []

    def add_candidate(raw_path: str) -> None:
        normalized = _normalize_gaia_relative_path(raw_path)
        if not normalized:
            return
        if normalized not in candidates:
            candidates.append(normalized)

    indexed_path = str(entry.get("file_path", "") or "").strip()
    indexed_name = _safe_filename(str(entry.get("file_name", "") or ""), "")
    safe_expected_name = _safe_filename(str(expected_filename or ""), "")

    if indexed_path:
        add_candidate(indexed_path)

    for name in (indexed_name, safe_expected_name):
        if not name:
            continue
        add_candidate(name)
        add_candidate(f"2023/validation/{name}")
        add_candidate(f"validation/{name}")

    preferred_ext = Path(safe_expected_name or indexed_name).suffix.lower().strip()
    safe_task_id = str(task_id or "").strip()
    if safe_task_id and preferred_ext:
        add_candidate(f"2023/validation/{safe_task_id}{preferred_ext}")

    if safe_task_id:
        for extension in _GAIA_ATTACHMENT_COMMON_EXTENSIONS:
            add_candidate(f"2023/validation/{safe_task_id}{extension}")

    return candidates


def _download_task_file_from_gaia_dataset(
    task_id: str,
    destination_dir: str | None = None,
    expected_filename: str = "",
) -> dict[str, Any]:
    safe_task_id = str(task_id or "").strip()
    if not safe_task_id:
        return {"status": "error", "message": "task_id is empty"}

    auth_headers = _hf_dataset_auth_headers()
    request_headers = dict(REQUEST_HEADERS)
    request_headers.update(auth_headers)

    entry, load_errors = _gaia_attachment_entry_for_task(safe_task_id, auth_headers=auth_headers)
    candidate_paths = _gaia_candidate_attachment_paths(
        task_id=safe_task_id,
        expected_filename=expected_filename,
        entry=entry,
    )

    if not candidate_paths:
        if load_errors:
            return {
                "status": "error",
                "message": f"GAIA dataset fallback index lookup failed ({load_errors})",
            }
        return {"status": "no_file", "message": "NO_FILE"}

    output_dir = Path(destination_dir) if destination_dir else _default_download_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    last_request_error = ""
    auth_required = False

    for relative_path in candidate_paths:
        resolved_path = relative_path.lstrip("/")
        url = f"{_GAIA_DATASET_RESOLVE_MAIN_URL}/{resolved_path}"

        try:
            response = requests.get(url, timeout=60, headers=request_headers)
        except requests.RequestException as err:
            last_request_error = str(err)
            continue

        if response.status_code in {401, 403}:
            auth_required = True
            continue

        if response.status_code == 404:
            continue

        try:
            response.raise_for_status()
        except requests.RequestException as err:
            last_request_error = str(err)
            continue

        filename = _safe_filename(Path(resolved_path).name, f"task_file_{safe_task_id}")
        destination = output_dir / filename
        with destination.open("wb") as handle:
            handle.write(response.content)

        return {
            "status": "downloaded",
            "path": str(destination.resolve()),
            "filename": filename,
            "message": "OK",
            "source": "gaia_dataset",
            "source_path": resolved_path,
        }

    if auth_required and not auth_headers.get("Authorization"):
        return {
            "status": "error",
            "message": (
                "GAIA dataset fallback requires a Hugging Face token. "
                "Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN)."
            ),
        }

    if auth_required:
        return {
            "status": "error",
            "message": (
                "GAIA dataset fallback access denied. Ensure the token has accepted "
                "gated dataset terms for gaia-benchmark/GAIA."
            ),
        }

    if last_request_error:
        return {
            "status": "error",
            "message": f"GAIA dataset fallback download failed ({last_request_error})",
        }

    return {"status": "no_file", "message": "NO_FILE"}


def download_task_file_raw(
    task_id: str,
    api_url: str | None = None,
    destination_dir: str | None = None,
    expected_filename: str = "",
) -> Dict[str, Any]:
    """Download task file with structured status for use in both graph and tools."""
    safe_task_id = (task_id or "").strip()
    if not safe_task_id:
        return {"status": "error", "message": "task_id is empty"}

    target_api = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{target_api}/files/{safe_task_id}"

    try:
        response = requests.get(url, timeout=60)
        if response.status_code in {401, 403, 404}:
            fallback_result = _download_task_file_from_gaia_dataset(
                task_id=safe_task_id,
                destination_dir=destination_dir,
                expected_filename=expected_filename,
            )

            if fallback_result.get("status") == "downloaded":
                return fallback_result

            if response.status_code == 404 and fallback_result.get("status") == "no_file":
                return {"status": "no_file", "message": "NO_FILE"}

            if fallback_result.get("status") in {"no_file", "error"}:
                return fallback_result

        response.raise_for_status()

        disposition = response.headers.get("content-disposition", "")
        default_name = f"task_file_{safe_task_id}"
        match = re.search(r"filename=(.+)", disposition)
        filename = _safe_filename(match.group(1), default_name) if match else default_name

        output_dir = Path(destination_dir) if destination_dir else _default_download_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / filename

        with destination.open("wb") as handle:
            handle.write(response.content)

        return {
            "status": "downloaded",
            "path": str(destination.resolve()),
            "filename": filename,
            "message": "OK",
        }
    except requests.RequestException as err:
        return {"status": "error", "message": f"failed to download file ({err})"}
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "message": f"unexpected download error ({err})"}


@tool
def download_task_file(task_id: str = "") -> str:
    """Download the file for a GAIA task_id and return absolute local file path."""
    safe_task_id = (task_id or "").strip() or get_runtime_task_id().strip()
    result = download_task_file_raw(
        task_id=safe_task_id,
        api_url=DEFAULT_API_URL,
        destination_dir=get_runtime_working_dir().strip() or None,
        expected_filename="",
    )

    if result.get("status") == "downloaded":
        return str(result.get("path"))
    if result.get("status") == "no_file":
        return "NO_FILE"
    return f"ERROR: {result.get('message', 'download failed')}"


@tool
def analyze_image_with_vlm(
    file_path: str,
    question: str,
    model_name: str = "",
    max_chars: int = 12000,
) -> str:
    """Analyze an image with Gemini VLM using the explicit user question."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        return f"ERROR: unsupported image format ({path.suffix.lower()})"

    prompt = (question or "").strip()
    if not prompt:
        return "ERROR: image question is empty"

    api_key = AgentConfig.from_env().google_api_key
    if not api_key:
        return "ERROR: GOOGLE_API_KEY is not configured"

    selected_model = (model_name or os.getenv("GAIA_VLM_MODEL", "gemini-2.5-flash")).strip() or "gemini-2.5-flash"

    try:
        payload_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime_type = _guess_image_mime_type(path)
        client = _get_vlm_client(selected_model, api_key)
        response = client.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{payload_b64}"},
                        },
                    ]
                )
            ]
        )
        text = _extract_text_from_message_content(getattr(response, "content", "")).strip()
        if not text:
            return "ERROR: VLM returned empty response"
        return _truncate(text, max_chars=max_chars)
    except Exception as err:  # noqa: BLE001
        return f"ERROR: VLM image analysis failed ({err})"


@tool
def web_search(query: str, max_results: int = 6, providers: str = "") -> str:
    """Search the web using concurrent providers (Tavily/Google/DDG) and return ranked URLs."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return "ERROR: empty search query"

    max_items = max(1, min(int(max_results), 10))
    config = AgentConfig.from_env()
    max_chars = config.max_web_results_chars

    provider_csv = str(providers or "").strip() or config.search_providers
    selected_providers = _resolve_search_providers(provider_csv)
    timeout_seconds = max(2, int(config.search_provider_timeout_seconds))

    hits_by_provider, provider_errors = _run_parallel_provider_search(
        query=cleaned_query,
        max_items=max_items,
        providers=selected_providers,
        timeout_seconds=timeout_seconds,
    )

    ranked_hits = _merge_and_rank_search_hits(hits_by_provider, max_items=max_items)
    if ranked_hits:
        rendered_hits = _format_search_hits_for_output(ranked_hits, max_chars=max_chars)
        warning_block = _format_provider_warnings(provider_errors)
        if warning_block:
            rendered_hits = _truncate(
                f"{rendered_hits}\n\n{warning_block}",
                max_chars=max_chars,
            )
        return rendered_hits

    # Safety net: always attempt DDG fallback once if caller did not request it.
    if "ddg" not in selected_providers:
        try:
            ddg_hits = _search_ddg_hits(
                query=cleaned_query,
                max_items=max_items,
                timeout_seconds=timeout_seconds,
            )
            ranked_fallback = _merge_and_rank_search_hits({"ddg": ddg_hits}, max_items=max_items)
            if ranked_fallback:
                return _format_search_hits_for_output(ranked_fallback, max_chars=max_chars)
        except Exception as err:  # noqa: BLE001
            provider_errors["ddg"] = f"{err.__class__.__name__}: {err}"

    if provider_errors:
        details = "; ".join(f"{name}={message}" for name, message in sorted(provider_errors.items()))
        return f"ERROR: web search failed ({details})"

    return "No search results found."


@tool
def arxiv_search(query: str, max_results: int = 3, max_chars: int = 12000) -> str:
    """Search arXiv papers and return concise metadata + snippets."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return "ERROR: empty arXiv query"

    load_max = max(1, min(int(max_results), 5))

    try:
        from langchain_community.document_loaders import ArxivLoader

        loader = ArxivLoader(query=cleaned_query, load_max_docs=load_max)
        docs = loader.load()
    except Exception as err:  # noqa: BLE001
        return f"ERROR: arXiv search failed ({err})"

    if not docs:
        return "No arXiv results found."

    lines: list[str] = []
    index = 1
    for doc in docs:
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        title = str(metadata.get("Title") or metadata.get("title") or "(no title)").strip()
        source_url = str(
            metadata.get("entry_id")
            or metadata.get("source")
            or metadata.get("url")
            or ""
        ).strip()
        if source_url and _is_potential_benchmark_leak_url(source_url):
            continue

        snippet = str(getattr(doc, "page_content", "") or "").strip()
        if not snippet:
            snippet = "(no abstract available)"

        if _contains_benchmark_leak_url_in_text(snippet) or _is_potential_benchmark_leak_content(snippet):
            continue

        published = str(metadata.get("Published") or metadata.get("published") or "").strip()
        authors = metadata.get("Authors") or metadata.get("authors") or []
        if isinstance(authors, list):
            authors_text = ", ".join(str(a).strip() for a in authors[:4] if str(a).strip())
        else:
            authors_text = str(authors).strip()

        lines.append(
            f"{index}. Title: {title}\n"
            f"URL: {source_url or '(no url)'}\n"
            f"Published: {published or '(unknown)'}\n"
            f"Authors: {authors_text or '(unknown)'}\n"
            f"Snippet: {_truncate(snippet, max_chars=1400)}"
        )
        index += 1

    if not lines:
        return "No safe arXiv results found after benchmark-leak filtering."

    return _truncate("\n\n".join(lines), max_chars=max_chars)


@tool
def fetch_webpage_text(url: str, max_chars: int = 12000) -> str:
    """Fetch a webpage URL and return cleaned visible text content."""
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        return "ERROR: empty URL"

    if _is_potential_benchmark_leak_url(cleaned_url):
        return (
            "ERROR: blocked potential benchmark-leak URL. "
            "Use primary sources (official pages/docs) or task attachments instead."
        )

    wiki_prefetch = _fetch_wikipedia_extract(cleaned_url, max_chars=max_chars)
    if wiki_prefetch:
        return wiki_prefetch

    try:
        response = requests.get(cleaned_url, timeout=45, headers=REQUEST_HEADERS)
        response.raise_for_status()
    except requests.RequestException as err:
        wiki_fallback = _fetch_wikipedia_extract(cleaned_url, max_chars=max_chars)
        if wiki_fallback:
            return wiki_fallback

        mirror_fallback = _fetch_with_jina_mirror(cleaned_url, max_chars=max_chars)
        if mirror_fallback:
            return mirror_fallback

        return f"ERROR: failed fetching webpage ({err})"

    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type:
        tmp_path = _default_download_dir() / "fetched_page.pdf"
        tmp_path.write_bytes(response.content)
        return extract_pdf_text.invoke({"file_path": str(tmp_path), "max_pages": 6})

    try:
        try:
            bs4_module = importlib.import_module("bs4")
            BeautifulSoup = getattr(bs4_module, "BeautifulSoup")
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()

            title = (soup.title.string or "").strip() if soup.title else ""
            body_text = " ".join(chunk.strip() for chunk in soup.stripped_strings if chunk.strip())
            body_text = _truncate(body_text, max_chars=max_chars)

            if title:
                rendered = f"Title: {title}\n\n{body_text}"
            else:
                rendered = body_text or "ERROR: no readable text found"

            if _is_potential_benchmark_leak_content(rendered):
                return (
                    "ERROR: blocked potential benchmark answer dump content. "
                    "Find evidence from primary sources instead."
                )

            return rendered
        except Exception:
            html_text = response.text
            title_match = re.search(r"<title>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
            cleaned = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
            cleaned = re.sub(r"<style[\s\S]*?</style>", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            cleaned = _truncate(cleaned, max_chars=max_chars)
            if title:
                rendered = f"Title: {title}\n\n{cleaned}"
            else:
                rendered = cleaned or "ERROR: no readable text found"

            if _is_potential_benchmark_leak_content(rendered):
                return (
                    "ERROR: blocked potential benchmark answer dump content. "
                    "Find evidence from primary sources instead."
                )

            return rendered
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed parsing webpage ({err})"


@tool
def get_youtube_video_context(url: str, max_chars: int = 5000) -> str:
    """Fetch title, description, and transcript hints for a YouTube video URL."""
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        return "ERROR: empty YouTube URL"

    video_id = _extract_youtube_video_id(cleaned_url)
    if not video_id:
        return "ERROR: invalid YouTube URL"

    lines: list[str] = [f"Video ID: {video_id}"]

    try:
        oembed = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": cleaned_url, "format": "json"},
            timeout=20,
        )
        if oembed.ok:
            payload = oembed.json()
            title = str(payload.get("title", "")).strip()
            author = str(payload.get("author_name", "")).strip()
            if title:
                lines.append(f"Title: {title}")
            if author:
                lines.append(f"Author: {author}")
    except Exception:  # noqa: BLE001
        pass

    try:
        page = requests.get(cleaned_url, timeout=30)
        page.raise_for_status()
        page_html = page.text
        desc_match = re.search(
            r'<meta\\s+name="description"\\s+content="([^"]+)"',
            page_html,
            flags=re.IGNORECASE,
        )
        if desc_match:
            lines.append(f"Description: {desc_match.group(1).strip()}")
    except Exception as err:  # noqa: BLE001
        lines.append(f"Description: unavailable ({err.__class__.__name__})")

    try:
        transcript_module = importlib.import_module("youtube_transcript_api")
        YouTubeTranscriptApi = getattr(transcript_module, "YouTubeTranscriptApi")
        transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join(str(chunk.get("text", "")).strip() for chunk in transcript_chunks)
        transcript_text = re.sub(r"\\s+", " ", transcript_text).strip()
        if transcript_text:
            lines.append(f"Transcript snippet: {_truncate(transcript_text, max_chars=max_chars)}")
        else:
            lines.append("Transcript snippet: unavailable (empty transcript)")
    except Exception as err:  # noqa: BLE001
        lines.append(f"Transcript snippet: unavailable ({err.__class__.__name__}: {err})")

    return "\n".join(lines)


@tool
def download_url_file(url: str, filename: str = "") -> str:
    """Download a file from any URL and return the local absolute path."""
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        return "ERROR: empty URL"

    lower_url = cleaned_url.lower()
    candidate_task_id = _extract_gaia_task_id(cleaned_url) or get_runtime_task_id().strip()
    parsed_url = urlparse(cleaned_url)
    candidate_expected_filename = _safe_filename(Path(parsed_url.path).name, "")

    # GAIA dataset attachment URLs may be inaccessible directly in some environments.
    if "huggingface.co/datasets/gaia-benchmark/gaia" in lower_url and candidate_task_id:
        fallback_result = download_task_file_raw(
            task_id=candidate_task_id,
            api_url=DEFAULT_API_URL,
            destination_dir=get_runtime_working_dir().strip() or None,
            expected_filename=candidate_expected_filename,
        )
        if fallback_result.get("status") == "downloaded":
            return str(fallback_result.get("path"))

    try:
        response = requests.get(cleaned_url, timeout=90)
        response.raise_for_status()

        chosen_name = (filename or "").strip()
        if not chosen_name:
            parsed = urlparse(cleaned_url)
            chosen_name = Path(parsed.path).name or "downloaded_file"

        output_dir = _default_download_dir()
        destination = output_dir / _safe_filename(chosen_name, "downloaded_file")
        with destination.open("wb") as handle:
            handle.write(response.content)

        return str(destination.resolve())
    except requests.RequestException as err:
        status_code = getattr(getattr(err, "response", None), "status_code", None)
        if status_code in {401, 403, 404} and candidate_task_id:
            fallback_result = download_task_file_raw(
                task_id=candidate_task_id,
                api_url=DEFAULT_API_URL,
                destination_dir=get_runtime_working_dir().strip() or None,
                expected_filename=candidate_expected_filename,
            )
            if fallback_result.get("status") == "downloaded":
                return str(fallback_result.get("path"))

        return f"ERROR: failed downloading URL ({err})"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: unexpected URL download error ({err})"


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a stateful interpreter and return output text."""
    return _python_executor.run(code)


@tool
def execute_bash(command: str) -> str:
    """Execute a short bash command (timeout-limited) in the task working directory."""
    return _bash_executor.run(command)


@tool
def read_text_file(file_path: str, max_chars: int = 12000) -> str:
    """Read a local text-like file and return its content (truncated for safety)."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _truncate(text, max_chars=max_chars)
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed reading file ({err})"


@tool
def extract_pdf_text(file_path: str, max_pages: int = 4) -> str:
    """Extract text from a PDF file for quick question answering."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages[: max(1, int(max_pages))], start=1):
            pages.append(f"--- Page {idx} ---\n{page.extract_text() or ''}")
        return "\n\n".join(pages).strip() or "ERROR: no extractable text in PDF"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed extracting PDF text ({err})"


@tool
def inspect_tabular_file(file_path: str, max_rows: int = 8) -> str:
    """Inspect CSV/XLS/XLSX and return schema + top rows as text."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    suffix = path.suffix.lower()
    try:
        import pandas as pd

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            return f"ERROR: unsupported tabular format ({suffix})"

        head = df.head(5)
        describe_text = ""
        try:
            describe_df = df.describe(include="all")
            describe_text = describe_df.to_string()
        except Exception:
            describe_text = "(describe unavailable for this table)"

        rendered = (
            f"Rows: {len(df)} | Columns: {len(df.columns)}\n"
            f"Column names: {', '.join(map(str, df.columns.tolist()))}\n\n"
            f"Preview (head 5):\n{head.to_string(index=False)}\n\n"
            f"Descriptive statistics:\n{describe_text}"
        )
        return _truncate(rendered, max_chars=12000)
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed reading tabular file ({err})"


@tool
def extract_office_text(file_path: str, max_chars: int = 12000) -> str:
    """Extract text from DOCX/PPTX files using ZIP/XML parsing."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    suffix = path.suffix.lower()
    if suffix not in {".docx", ".pptx"}:
        return f"ERROR: unsupported office format ({suffix})"

    try:
        with zipfile.ZipFile(path) as archive:
            if suffix == ".docx":
                targets = ["word/document.xml"]
            else:
                targets = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                )

            if not targets:
                return "ERROR: no extractable text in office file"

            chunks: list[str] = []
            for target in targets:
                try:
                    xml_blob = archive.read(target).decode("utf-8", errors="ignore")
                except KeyError:
                    continue
                chunks.append(_strip_xml_text(xml_blob))

        text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
        if not text:
            return "ERROR: no extractable text in office file"

        return _truncate(text, max_chars=max_chars)
    except zipfile.BadZipFile as err:
        return f"ERROR: invalid office file ({err})"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed extracting office text ({err})"


@tool
def inspect_archive_file(file_path: str, max_entries: int = 40) -> str:
    """Inspect ZIP archives and return entry names/sizes."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    if path.suffix.lower() != ".zip":
        return f"ERROR: unsupported archive format ({path.suffix.lower()})"

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos:
                return "Archive is empty"

            limit = max(1, int(max_entries))
            lines = [f"Archive entries: {len(infos)}"]
            for info in infos[:limit]:
                lines.append(f"- {info.filename} ({info.file_size} bytes)")

            if len(infos) > limit:
                lines.append(f"... and {len(infos) - limit} more entries")

            return "\n".join(lines)
    except zipfile.BadZipFile as err:
        return f"ERROR: invalid ZIP archive ({err})"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed reading archive ({err})"


@tool
def transcribe_audio_file(file_path: str, max_chars: int = 12000) -> str:
    """Transcribe local audio files (mp3/wav/m4a/flac/ogg/aac) to text."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    suffix = path.suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        return f"ERROR: unsupported audio format ({suffix})"

    try:
        transcriber = _get_audio_transcriber()
        result = transcriber(str(path))
        if isinstance(result, dict):
            text = str(result.get("text", "")).strip()
        else:
            text = str(result).strip()

        if not text:
            return "ERROR: no speech detected in audio file"
        return _truncate(text, max_chars=max_chars)
    except Exception as err:  # noqa: BLE001
        return (
            "ERROR: audio transcription failed "
            f"({err}). Ensure audio dependencies are available and retry."
        )


@tool
def ocr_image_file(file_path: str) -> str:
    """Extract text from an image file using OCR."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    try:
        import pytesseract
        from PIL import Image

        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        text = text.strip()
        if not text:
            return "ERROR: no text detected in image"
        return text
    except Exception as err:  # noqa: BLE001
        return f"ERROR: OCR failed ({err})"


@tool
def list_working_directory() -> str:
    """List files currently available in the active GAIA working directory."""
    base = _default_download_dir()
    items = sorted(base.iterdir())
    if not items:
        return f"No files found in {base}"

    lines = [f"Working directory: {base}"]
    for item in items:
        suffix = "/" if item.is_dir() else ""
        lines.append(f"- {item.name}{suffix}")
    return "\n".join(lines)


@tool
def inspect_local_file(file_path: str) -> str:
    """Inspect a local file by extension and return a useful textual summary."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"ERROR: file not found ({path})"

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".py", ".yaml", ".yml", ".xml", ".html", ".htm"}:
        return read_text_file.invoke({"file_path": str(path), "max_chars": 12000})
    if suffix == ".csv":
        return inspect_tabular_file.invoke({"file_path": str(path), "max_rows": 10})
    if suffix == ".pdf":
        return extract_pdf_text.invoke({"file_path": str(path), "max_pages": 5})
    if suffix in {".xlsx", ".xls"}:
        return inspect_tabular_file.invoke({"file_path": str(path), "max_rows": 10})
    if suffix in {".docx", ".pptx"}:
        return extract_office_text.invoke({"file_path": str(path), "max_chars": 12000})
    if suffix == ".zip":
        return inspect_archive_file.invoke({"file_path": str(path), "max_entries": 40})
    if suffix in _AUDIO_SUFFIXES:
        return transcribe_audio_file.invoke({"file_path": str(path), "max_chars": 12000})
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        return ocr_image_file.invoke({"file_path": str(path)})

    try:
        raw = path.read_bytes()[:256]
        return f"Binary file: {path.name} ({path.stat().st_size} bytes). First 256 bytes: {raw!r}"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed to inspect file ({err})"


def get_tools(public_mode: bool | None = None, allow_unsafe_tools: bool | None = None) -> List:
    """Return the default tool set for the GAIA LangGraph agent."""
    config = AgentConfig.from_env()
    resolved_public_mode = config.web_public_mode if public_mode is None else bool(public_mode)
    resolved_allow_unsafe = (
        config.allow_unsafe_tools if allow_unsafe_tools is None else bool(allow_unsafe_tools)
    )

    tools: List = [
        download_task_file,
        analyze_image_with_vlm,
        web_search,
        arxiv_search,
        fetch_webpage_text,
        fetch_wikipedia_section,
        get_youtube_video_context,
        download_url_file,
    ]

    include_unsafe_tools = (not resolved_public_mode) or resolved_allow_unsafe
    if include_unsafe_tools:
        tools.extend([execute_python, execute_bash])

    tools.extend(
        [
            read_text_file,
            extract_pdf_text,
            inspect_tabular_file,
            extract_office_text,
            inspect_archive_file,
            transcribe_audio_file,
            ocr_image_file,
            list_working_directory,
            inspect_local_file,
        ]
    )
    return tools
