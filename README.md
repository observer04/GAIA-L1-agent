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
2. Install dependencies from `requirements.txt`.
3. Run the app locally for end-to-end checks.
4. Use LangGraph Studio with `langgraph.json` to inspect multi-turn traces and tool routing.

### Notes

- Submission answers are normalized to avoid prefix/wrapper mismatches.
- `tools.py` remains as a compatibility re-export layer to avoid breaking existing imports/tests.
