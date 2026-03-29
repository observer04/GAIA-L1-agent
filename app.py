import os
import json
import threading
from pathlib import Path
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import gradio as gr
import pandas as pd
import requests

from agent import GaiaLangGraphAgent, normalize_answer
from agent.config import AgentConfig


SUBMISSION_CACHE_DIR = Path("artifacts/submission_cache")
_THREAD_LOCAL = threading.local()


def _fetch_questions(api_url: str) -> list[dict]:
    response = requests.get(f"{api_url.rstrip('/')}/questions", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Questions response is not a list.")
    return payload


def _submit_answers(api_url: str, submission_data: dict) -> dict:
    response = requests.post(
        f"{api_url.rstrip('/')}/submit",
        json=submission_data,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def _safe_username(username: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in username.strip().lower())


def _get_agent_code_url() -> str:
    space_id = os.getenv("SPACE_ID", "")
    if space_id:
        return f"https://huggingface.co/spaces/{space_id}/tree/main"
    return "https://huggingface.co/spaces/<your-space>/tree/main"


def _cache_path_for_user(username: str) -> Path:
    SUBMISSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_user = _safe_username(username)
    return SUBMISSION_CACHE_DIR / f"submission_cache_{safe_user}_{timestamp}.json"


def _latest_cache_for_user(username: str) -> Path | None:
    SUBMISSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_user = _safe_username(username)
    files = sorted(SUBMISSION_CACHE_DIR.glob(f"submission_cache_{safe_user}_*.json"))
    if not files:
        return None
    return files[-1]


def _save_submission_cache(payload: dict[str, Any]) -> Path:
    username = str(payload.get("username", "anonymous")).strip() or "anonymous"
    path = _cache_path_for_user(username)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_submission_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _submission_max_workers() -> int:
    raw = os.getenv("GAIA_SUBMISSION_MAX_WORKERS", "2")
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 2
    return max(1, min(parsed, 4))


def _thread_agent(config: AgentConfig) -> GaiaLangGraphAgent:
    current = getattr(_THREAD_LOCAL, "agent", None)
    current_key = getattr(_THREAD_LOCAL, "agent_key", None)
    key = (config.model_name, config.api_url, config.timeout_seconds, bool(config.google_api_key))
    if current is None or current_key != key:
        current = GaiaLangGraphAgent(config=config)
        _THREAD_LOCAL.agent = current
        _THREAD_LOCAL.agent_key = key
    return current


def _run_task_with_thread_agent(
    config: AgentConfig,
    task_id: str,
    question: str,
    level: str = "",
    file_name: str = "",
) -> dict[str, Any]:
    agent = _thread_agent(config)
    return agent.run_task(
        question=question,
        task_id=task_id,
        run_label="submission",
        level=level,
        file_name=file_name,
    )


def _generate_answers(
    config: AgentConfig,
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    entries: list[tuple[int, str, str, str, str]] = []
    for idx, item in enumerate(questions):
        task_id = str(item.get("task_id", "")).strip()
        question = str(item.get("question", "")).strip()
        level = str(item.get("Level", item.get("level", ""))).strip()
        file_name = str(item.get("file_name", "")).strip()
        if task_id and question:
            entries.append((idx, task_id, question, level, file_name))

    if not entries:
        return [], []

    answers_by_index: dict[int, dict[str, str]] = {}
    logs_by_index: dict[int, dict[str, Any]] = {}

    worker_count = _submission_max_workers()
    if worker_count == 1:
        for idx, task_id, question, level, file_name in entries:
            run_result = _run_task_with_thread_agent(
                config=config,
                task_id=task_id,
                question=question,
                level=level,
                file_name=file_name,
            )
            submitted_answer = normalize_answer(run_result.get("submitted_answer", "I don't know"))
            answers_by_index[idx] = {"task_id": task_id, "submitted_answer": submitted_answer}
            logs_by_index[idx] = {
                "Task ID": task_id,
                "Submitted Answer": submitted_answer,
                "Level": level,
                "Latency (s)": run_result.get("latency_seconds", ""),
                "Attempts": run_result.get("attempt_count", ""),
                "Stop Reason": run_result.get("stop_reason", ""),
                "Task File Status": run_result.get("task_file_status", ""),
                "Status": run_result.get("status", ""),
            }
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _run_task_with_thread_agent,
                    config,
                    task_id,
                    question,
                    level,
                    file_name,
                ): (idx, task_id, level)
                for idx, task_id, question, level, file_name in entries
            }

            for future in as_completed(future_map):
                idx, task_id, level = future_map[future]
                try:
                    run_result = future.result()
                except Exception as err:  # noqa: BLE001
                    run_result = {
                        "status": "error",
                        "submitted_answer": "I don't know",
                        "latency_seconds": "",
                        "attempt_count": 0,
                        "stop_reason": f"thread_exception: {err}",
                        "task_file_status": "error",
                    }

                submitted_answer = normalize_answer(run_result.get("submitted_answer", "I don't know"))
                answers_by_index[idx] = {"task_id": task_id, "submitted_answer": submitted_answer}
                logs_by_index[idx] = {
                    "Task ID": task_id,
                    "Submitted Answer": submitted_answer,
                    "Level": level,
                    "Latency (s)": run_result.get("latency_seconds", ""),
                    "Attempts": run_result.get("attempt_count", ""),
                    "Stop Reason": run_result.get("stop_reason", ""),
                    "Task File Status": run_result.get("task_file_status", ""),
                    "Status": run_result.get("status", ""),
                }

    ordered_indexes = sorted(answers_by_index.keys())
    answers_payload = [answers_by_index[idx] for idx in ordered_indexes]
    results_log = [logs_by_index[idx] for idx in ordered_indexes]
    return answers_payload, results_log


def generate_answers_and_cache(profile: gr.OAuthProfile | None):
    """Generate answers, cache them to disk, and return run logs without submitting."""
    if not profile:
        return "Please login to Hugging Face first.", "", None

    username = profile.username.strip()
    config = AgentConfig.from_env()

    try:
        _ = GaiaLangGraphAgent(config=config)
    except Exception as err:  # noqa: BLE001
        return f"Error initializing agent: {err}", "", None

    try:
        questions = _fetch_questions(config.api_url)
    except Exception as err:  # noqa: BLE001
        return f"Error fetching questions: {err}", "", None

    answers_payload, results_log = _generate_answers(config=config, questions=questions)
    if not answers_payload:
        return "Agent did not generate any answers.", "", pd.DataFrame(results_log)

    cache_payload: dict[str, Any] = {
        "username": username,
        "agent_code": _get_agent_code_url(),
        "generated_at": datetime.now(UTC).isoformat(),
        "worker_count": _submission_max_workers(),
        "question_count": len(answers_payload),
        "answers": answers_payload,
        "results_log": results_log,
    }
    cache_path = _save_submission_cache(cache_payload)

    status = (
        "Answer generation complete.\n"
        f"User: {username}\n"
        f"Questions answered: {len(answers_payload)}\n"
        f"Workers used: {_submission_max_workers()}\n"
        f"Cache file: {cache_path}"
    )
    return status, str(cache_path), pd.DataFrame(results_log)


def submit_cached_answers(profile: gr.OAuthProfile | None):
    """Submit the latest cached answers for the logged-in user."""
    if not profile:
        return "Please login to Hugging Face first."

    username = profile.username.strip()
    config = AgentConfig.from_env()

    cache_path = _latest_cache_for_user(username)
    if cache_path is None:
        return "No cached answers found. Click 'Generate Answers (Cache Only)' first."

    try:
        cache_payload = _load_submission_cache(cache_path)
    except Exception as err:  # noqa: BLE001
        return f"Failed to load cache file {cache_path}: {err}"

    answers_payload = cache_payload.get("answers", [])
    if not isinstance(answers_payload, list) or not answers_payload:
        return f"Cache file has no answers: {cache_path}"

    submission_data = {
        "username": username,
        "agent_code": _get_agent_code_url(),
        "answers": answers_payload,
    }

    try:
        result_data = _submit_answers(config.api_url, submission_data)
        return (
            "Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}\n"
            f"Submitted cache: {cache_path}"
        )
    except requests.exceptions.HTTPError as err:
        details = f"HTTP {err.response.status_code}"
        try:
            payload = err.response.json()
            details += f" - {payload.get('detail', '')}"
        except Exception:  # noqa: BLE001
            details += f" - {err.response.text[:500]}"
        return f"Submission Failed: {details}\nCache: {cache_path}"
    except Exception as err:  # noqa: BLE001
        return f"Submission Failed: {err}\nCache: {cache_path}"


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner (LangGraph v3)")
    gr.Markdown(
        """
        1. Login with your Hugging Face account.
        2. Click **Generate Answers (Cache Only)** to answer all questions and cache results locally.
        3. Review logs and cached file path.
        4. Click **Submit Cached Answers** as a separate action.

        Notes:
        - The benchmark checks exact-match outputs.
        - Answers are normalized before cache/submit.
        - Generation supports threaded execution via `GAIA_SUBMISSION_MAX_WORKERS` (default: 2, max: 4).
        """
    )

    gr.LoginButton()
    generate_button = gr.Button("Generate Answers (Cache Only)", variant="primary")
    submit_button = gr.Button("Submit Cached Answers")

    generation_status_output = gr.Textbox(label="Generation Status", lines=6, interactive=False)
    cache_path_output = gr.Textbox(label="Latest Cache File", lines=2, interactive=False)
    submit_status_output = gr.Textbox(label="Submission Status", lines=6, interactive=False)
    results_table = gr.DataFrame(label="Per-task Run Logs", wrap=True)

    generate_button.click(
        fn=generate_answers_and_cache,
        outputs=[generation_status_output, cache_path_output, results_table],
    )

    submit_button.click(
        fn=submit_cached_answers,
        outputs=[submit_status_output],
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
