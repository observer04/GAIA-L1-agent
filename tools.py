"""Compatibility facade for legacy imports.

The v2 agent implementation lives under `agent.tools`, but this module is kept to
avoid breaking imports in existing scripts/tests.
"""

from agent.tools import (  # noqa: F401
    DEFAULT_API_URL,
    download_task_file,
    download_url_file,
    execute_python,
    extract_pdf_text,
    get_tools,
    inspect_local_file,
    inspect_tabular_file,
    list_working_directory,
    ocr_image_file,
    read_text_file,
    web_search,
)

__all__ = [
    "DEFAULT_API_URL",
    "download_task_file",
    "download_url_file",
    "web_search",
    "execute_python",
    "read_text_file",
    "extract_pdf_text",
    "inspect_tabular_file",
    "ocr_image_file",
    "list_working_directory",
    "inspect_local_file",
    "get_tools",
]
