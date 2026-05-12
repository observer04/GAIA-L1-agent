---
title: GAIA LangGraph v2 Agent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA LangGraph v2 Agent

Production-style GAIA benchmark agent built on LangGraph with Gemini, a robust tool suite, and safety-focused evidence handling.

## Overview

This repository provides a modular GAIA Level-1 agent that emphasizes tool-grounded reasoning, attachment-aware workflows, and exact-match answer formatting. It ships with a Gradio UI for submissions, a fast CLI runner for concurrent generation, and local evaluation utilities.

## Key capabilities

- **LangGraph orchestration**: staged workflow with context initialization, task-file retrieval, tool execution, evidence checks, replanning, and finalization.
- **Multi-provider web search**: parallelized search across Tavily, Google, and DuckDuckGo with ranking, deduplication, and leak filtering.
- **Attachment-first reasoning**: automatic download of GAIA task files with dataset fallback and per-task working directories.
- **Multimodal analysis**: image analysis via Gemini VLM, OCR for images, and audio transcription support.
- **Structured file inspection**: PDF, Office, tabular, archive, text, and generic file summaries.
- **Computation tools**: stateful Python execution plus short bash commands for local file operations.
- **Answer normalization**: strict exact-match formatting and robust post-processing.
- **Submission resiliency**: cached two-step submission flow with optional concurrency.

## Workflow architecture

1. **Context initialization** sets task metadata and working directories.
2. **Task file fetch** retrieves attachments when required.
3. **Assistant + tools loop** invokes the tool suite through LangGraph.
4. **Evidence checks** detect stagnation, enforce iteration budgets, and trigger replanning.
5. **Finalization** synthesizes the best-supported answer and enforces strict output format.

## Tooling overview

**Search & browsing**
- `web_search` (Tavily/Google/DDG providers with ranking and leak safeguards)
- `fetch_webpage_text` and `fetch_wikipedia_section`
- `arxiv_search`
- `get_youtube_video_context`

**Attachments & files**
- `download_task_file` and `download_url_file`
- `read_text_file`, `extract_pdf_text`
- `inspect_tabular_file` (CSV/XLS/XLSX)
- `extract_office_text` (DOCX/PPTX)
- `inspect_archive_file` (ZIP)
- `inspect_local_file`, `list_working_directory`

**Multimodal**
- `analyze_image_with_vlm`
- `ocr_image_file`
- `transcribe_audio_file`

**Execution**
- `execute_python` (stateful)
- `execute_bash` (timeout-limited)

## Quick start (local)

1. Set required environment variables (see configuration below).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the Gradio app: `python app.py`.
4. Optional: open LangGraph Studio using `langgraph.json` for tracing and debugging.

## Configuration

**Required**
- `GOOGLE_API_KEY`: Gemini API key for LLM + VLM calls.

**Model tuning**
- `GEMINI_MODEL` (default: `gemini-3-flash-preview`)
- `GEMINI_TEMPERATURE`
- `AGENT_TIMEOUT_SECONDS`, `AGENT_MAX_ITERATIONS`, `AGENT_LEVEL1_MAX_ITERATIONS`
- `GAIA_VLM_MODEL` (default: `gemini-2.5-flash`)

**Search providers**
- `SEARCH_PROVIDERS` (default: `tavily,google,ddg`)
- `SEARCH_PROVIDER_TIMEOUT_SECONDS`
- `TAVILY_API_KEY`, `TAVILY_SEARCH_DEPTH`, `TAVILY_INCLUDE_ANSWER`, `TAVILY_INCLUDE_RAW_CONTENT`
- `GOOGLE_SEARCH_API_KEY`, `GOOGLE_CSE_ID` (fallback Google CSE)

**Runtime & tracing**
- `GAIA_API_URL`
- `GAIA_WORKING_DIR`
- `LANGSMITH_TRACING` or `LANGCHAIN_TRACING_V2`
- `LANGSMITH_PROJECT`

**Submission & caching**
- `GAIA_SUBMISSION_MAX_WORKERS` (UI, max 4)
- `GAIA_FAST_SUBMISSION_MAX_WORKERS` (CLI, max 20)
- `GAIA_FAST_TASK_TIMEOUT_SECONDS`
- `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` (attachment fallback and username discovery)
- `SPACE_ID` (used to construct agent code URL for submissions)

## Submission flow (UI)

The Gradio UI uses a two-step process:
1. **Generate Answers (Cache Only)** — runs the agent across all questions and stores answers under `artifacts/submission_cache/`.
2. **Submit Cached Answers** — uploads the latest cache without rerunning the benchmark.

This separates long-running generation from the submission step and supports threaded generation.

## CLI utilities

- **Fast concurrent submission**: `fast_submit.py` mirrors the cache/submit schema and supports multi-worker generation.
  - Cache + submit: `python fast_submit.py --workers 8`
  - Cache only: `python fast_submit.py --workers 8 --cache-only`
  - Submit latest cache: `python fast_submit.py --submit-only`
- **Local evaluation**: `local_eval.py` supports random, sample, full, and focused runs, producing JSON reports under `artifacts/local_eval/`.

## Project structure

- `app.py`: Gradio UI + submission workflow.
- `agent/graph.py`: LangGraph state machine and orchestration.
- `agent/tools/core.py`: tool implementations (web, files, multimodal, execution).
- `fast_submit.py`: high-concurrency CLI runner.
- `local_eval.py`: local evaluation harness.
- `langgraph.json`: LangGraph Studio config.
- `tools.py`: compatibility re-export for legacy imports.

## Testing

Run tests with `python -m pytest`. (Requires `pytest` to be installed in your environment.)
