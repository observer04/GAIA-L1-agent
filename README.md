---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at <https://huggingface.co/docs/hub/spaces-config-reference>

## GAIA LangGraph v2 (in progress)

This branch starts a modular LangGraph-based GAIA agent implementation.

### Main entrypoints

- `app.py`: Gradio submission UI and API loop.
- `agent/graph.py`: staged LangGraph workflow and agent wrapper.
- `agent/tools/core.py`: hardened tool suite (web, download, python, pdf/text/tabular/image).
- `langgraph.json`: local LangGraph Studio configuration.

### Local validation

1. Set required environment variables in `.env`:
   - `GOOGLE_API_KEY=...`
   - optional: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...`, `LANGSMITH_PROJECT=gaia-agent-v2`
   - optional tuning: `AGENT_MAX_ITERATIONS=12`, `AGENT_LEVEL1_MAX_ITERATIONS=8` (for tighter Level-1 loops), `GEMINI_TEMPERATURE=1.0`
   - optional concurrent search routing:
     - `SEARCH_PROVIDERS=tavily,google,ddg`
     - `SEARCH_PROVIDER_TIMEOUT_SECONDS=8`
     - `TAVILY_API_KEY=...`
       - optional Tavily tuning: `TAVILY_SEARCH_DEPTH=basic|advanced`, `TAVILY_INCLUDE_ANSWER=true|false`, `TAVILY_INCLUDE_RAW_CONTENT=true|false`
       - Google provider (primary path): `GOOGLE_API_KEY=...` (uses Gemini Google Search tool grounding)
       - optional Google fallback path: `GOOGLE_SEARCH_API_KEY=...`, `GOOGLE_CSE_ID=...` (Custom Search API)
   - optional attachment fallback for `/files/{task_id}` outages:
     - `HF_TOKEN=...` (or `HUGGINGFACEHUB_API_TOKEN=...`) with accepted access to `gaia-benchmark/GAIA`
2. Install dependencies from `requirements.txt`.
3. Run the app locally for end-to-end checks.
4. Use LangGraph Studio with `langgraph.json` to inspect multi-turn traces and tool routing.

### Notes

- Submission answers are normalized to avoid prefix/wrapper mismatches.
- `tools.py` remains as a compatibility re-export layer to avoid breaking existing imports/tests.

### Submission flow (delay mitigation)

The UI now uses a two-step submission process:

1. **Generate Answers (Cache Only)**
   - Runs the agent across all benchmark questions.
   - Stores generated answers in `artifacts/submission_cache/`.
2. **Submit Cached Answers**
   - Loads the latest cache for the logged-in user.
   - Submits in a separate action.

This avoids rerunning the whole benchmark every time submit is clicked and makes long runs more manageable.

Optional async acceleration:

- Set `GAIA_SUBMISSION_MAX_WORKERS` (default `2`, max `4`) to control threaded answer generation.

### Fast concurrent CLI submission

When local Gradio generation is too slow, use the CLI runner that mirrors the same cache/submit schema (`username`, `agent_code`, `answers`) and writes into `artifacts/submission_cache/`.

- Generate + submit (concurrent): `python fast_submit.py --workers 8`
- Generate cache only: `python fast_submit.py --workers 8 --cache-only`
- Submit latest cached answers for your username: `python fast_submit.py --submit-only`

Notes:

- Username is resolved from `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` if `--username` is omitted.
- Resume is enabled by default and reruns cached unusable answers (e.g., `I don't know`) automatically.
- To keep prior IDK answers during resume, pass `--keep-idk-resume`.
- To prevent long hangs on a single task, use `--task-timeout-seconds` (default: `1200`).
- You can tune worker count with `--workers` or env var `GAIA_FAST_SUBMISSION_MAX_WORKERS`.
