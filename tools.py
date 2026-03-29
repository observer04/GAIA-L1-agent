"""Compatibility facade for legacy imports.

The v2 agent implementation lives under `agent.tools`, but this module is kept to
avoid breaking imports in existing scripts/tests.
"""

from agent.tools import (  # noqa: F401
    DEFAULT_API_URL,
    analyze_image_with_vlm,
    arxiv_search,
    download_task_file,
    download_url_file,
    execute_bash,
    execute_python,
    extract_pdf_text,
    extract_office_text,
    get_tools,
    fetch_webpage_text,
    fetch_wikipedia_section,
    get_youtube_video_context,
    inspect_archive_file,
    inspect_local_file,
    inspect_tabular_file,
    list_working_directory,
    ocr_image_file,
    read_text_file,
    transcribe_audio_file,
    web_search,
)

__all__ = [
    "DEFAULT_API_URL",
    "analyze_image_with_vlm",
    "arxiv_search",
    "download_task_file",
    "download_url_file",
    "web_search",
    "fetch_webpage_text",
    "fetch_wikipedia_section",
    "get_youtube_video_context",
    "execute_python",
    "execute_bash",
    "read_text_file",
    "extract_pdf_text",
    "extract_office_text",
    "inspect_tabular_file",
    "inspect_archive_file",
    "transcribe_audio_file",
    "ocr_image_file",
    "list_working_directory",
    "inspect_local_file",
    "get_tools",
]
