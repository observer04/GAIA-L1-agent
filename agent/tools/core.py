from __future__ import annotations

import os
import re
import tempfile
import importlib
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, List

import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from ..config import AgentConfig
from .python_executor import StatefulPythonExecutor
from .runtime import get_runtime_task_id, get_runtime_working_dir

DEFAULT_API_URL = AgentConfig.from_env().api_url

_python_executor = StatefulPythonExecutor()


def _safe_filename(raw_name: str, fallback: str) -> str:
    name = (raw_name or "").strip().strip("\"'")
    if not name:
        return fallback
    return os.path.basename(name)


def _truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


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


def download_task_file_raw(
    task_id: str,
    api_url: str | None = None,
    destination_dir: str | None = None,
) -> Dict[str, Any]:
    """Download task file with structured status for use in both graph and tools."""
    safe_task_id = (task_id or "").strip()
    if not safe_task_id:
        return {"status": "error", "message": "task_id is empty"}

    target_api = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{target_api}/files/{safe_task_id}"

    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            return {"status": "no_file", "message": "NO_FILE"}
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
    )

    if result.get("status") == "downloaded":
        return str(result.get("path"))
    if result.get("status") == "no_file":
        return "NO_FILE"
    return f"ERROR: {result.get('message', 'download failed')}"


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo and return concise textual results."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return "ERROR: empty search query"

    try:
        ddg_search = DuckDuckGoSearchRun()
        raw_result = str(ddg_search.run(cleaned_query))
        max_chars = AgentConfig.from_env().max_web_results_chars
        if len(raw_result) > max_chars:
            return raw_result[:max_chars] + "..."
        return raw_result
    except Exception as err:  # noqa: BLE001
        return f"ERROR: web search failed ({err})"


@tool
def fetch_webpage_text(url: str, max_chars: int = 12000) -> str:
    """Fetch a webpage URL and return cleaned visible text content."""
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        return "ERROR: empty URL"

    try:
        response = requests.get(cleaned_url, timeout=45)
        response.raise_for_status()
    except requests.RequestException as err:
        return f"ERROR: failed fetching webpage ({err})"

    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type:
        tmp_path = _default_download_dir() / "fetched_page.pdf"
        tmp_path.write_bytes(response.content)
        return extract_pdf_text.invoke({"file_path": str(tmp_path), "max_pages": 6})

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
            return f"Title: {title}\n\n{body_text}"
        return body_text or "ERROR: no readable text found"
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
        html = page.text
        desc_match = re.search(
            r'<meta\\s+name="description"\\s+content="([^"]+)"',
            html,
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
        return f"ERROR: failed downloading URL ({err})"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: unexpected URL download error ({err})"


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a stateful interpreter and return output text."""
    return _python_executor.run(code)


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

        head = df.head(max(1, int(max_rows)))
        return (
            f"Rows: {len(df)} | Columns: {len(df.columns)}\n"
            f"Column names: {', '.join(map(str, df.columns.tolist()))}\n\n"
            f"Preview:\n{head.to_string(index=False)}"
        )
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed reading tabular file ({err})"


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
    if suffix in {".txt", ".md", ".json", ".py", ".csv"}:
        return read_text_file.invoke({"file_path": str(path), "max_chars": 12000})
    if suffix == ".pdf":
        return extract_pdf_text.invoke({"file_path": str(path), "max_pages": 5})
    if suffix in {".xlsx", ".xls"}:
        return inspect_tabular_file.invoke({"file_path": str(path), "max_rows": 10})
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        return ocr_image_file.invoke({"file_path": str(path)})

    try:
        raw = path.read_bytes()[:256]
        return f"Binary file: {path.name} ({path.stat().st_size} bytes). First 256 bytes: {raw!r}"
    except Exception as err:  # noqa: BLE001
        return f"ERROR: failed to inspect file ({err})"


def get_tools() -> List:
    """Return the default tool set for the GAIA LangGraph agent."""
    return [
        download_task_file,
        web_search,
        fetch_webpage_text,
        get_youtube_video_context,
        download_url_file,
        execute_python,
        read_text_file,
        extract_pdf_text,
        inspect_tabular_file,
        ocr_image_file,
        list_working_directory,
        inspect_local_file,
    ]
