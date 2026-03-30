from __future__ import annotations

import argparse
import json
import os
import threading
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from agent import GaiaLangGraphAgent, normalize_answer
from agent.config import AgentConfig
from agent.postprocess import is_unusable_answer


SUBMISSION_CACHE_DIR = Path("artifacts/submission_cache")
_THREAD_LOCAL = threading.local()
_HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _default_task_timeout_seconds() -> int:
    raw_value = str(os.getenv("GAIA_FAST_TASK_TIMEOUT_SECONDS", "1200") or "").strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        parsed = 1200
    return max(0, parsed)


def _fetch_questions(api_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{api_url.rstrip('/')}/questions", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Questions response is not a list.")
    return payload


def _submit_answers(api_url: str, submission_data: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/submit",
        json=submission_data,
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Submission response is not an object.")
    return payload


def _safe_username(username: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in username.strip().lower())


def _hf_token_from_env() -> str:
    for key in _HF_TOKEN_ENV_KEYS:
        token = str(os.getenv(key, "") or "").strip()
        if token:
            return token
    return ""


def _resolve_username(explicit_username: str) -> str:
    provided = str(explicit_username or "").strip()
    if provided:
        return provided

    token = _hf_token_from_env()
    if not token:
        raise ValueError(
            "Username not provided and no Hugging Face token found. "
            "Use --username or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN."
        )

    response = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    response.raise_for_status()

    payload_raw = response.json()
    payload = payload_raw if isinstance(payload_raw, dict) else {}

    username = str(payload.get("name", "") or "").strip()
    if not username:
        auth_info = payload.get("auth")
        if isinstance(auth_info, dict):
            username = str(auth_info.get("name", "") or auth_info.get("user", "") or "").strip()

    if not username:
        raise ValueError("Unable to resolve username from Hugging Face whoami response.")

    return username


def _get_agent_code_url() -> str:
    space_id = os.getenv("SPACE_ID", "")
    if space_id:
        return f"https://huggingface.co/spaces/{space_id}/tree/main"
    return "https://huggingface.co/spaces/<your-space>/tree/main"


def _cache_path_for_user(username: str) -> Path:
    SUBMISSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_user = _safe_username(username)
    return SUBMISSION_CACHE_DIR / f"submission_cache_{safe_user}_{timestamp}.json"


def _latest_cache_for_user(username: str) -> Path | None:
    SUBMISSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_user = _safe_username(username)
    files = sorted(SUBMISSION_CACHE_DIR.glob(f"submission_cache_{safe_user}_*.json"))
    if not files:
        return None
    return files[-1]


def _write_submission_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_submission_cache(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cache payload at {path} is not an object")
    return payload


def _default_workers() -> int:
    for env_key in ("GAIA_FAST_SUBMISSION_MAX_WORKERS", "GAIA_SUBMISSION_MAX_WORKERS"):
        raw_value = str(os.getenv(env_key, "") or "").strip()
        if not raw_value:
            continue
        try:
            return max(1, min(int(raw_value), 20))
        except ValueError:
            continue
    return 8


def _thread_agent(config: AgentConfig) -> GaiaLangGraphAgent:
    current = getattr(_THREAD_LOCAL, "agent", None)
    current_key = getattr(_THREAD_LOCAL, "agent_key", None)
    key = (
        config.model_name,
        config.api_url,
        config.timeout_seconds,
        bool(config.google_api_key),
        config.max_iterations,
        config.level1_max_iterations,
    )
    if current is None or current_key != key:
        current = GaiaLangGraphAgent(config=config)
        _THREAD_LOCAL.agent = current
        _THREAD_LOCAL.agent_key = key
    return current


def _run_task_with_thread_agent(config: AgentConfig, entry: dict[str, Any]) -> dict[str, Any]:
    agent = _thread_agent(config)
    return agent.run_task(
        question=str(entry.get("question", "")),
        task_id=str(entry.get("task_id", "")),
        run_label="submission",
        level=str(entry.get("level", "")),
        file_name=str(entry.get("file_name", "")),
    )


def _build_error_run_result(
    stop_reason: str,
    latency_seconds: float,
    task_file_status: str = "error",
) -> dict[str, Any]:
    return {
        "status": "error",
        "submitted_answer": "I don't know",
        "latency_seconds": round(float(latency_seconds), 3),
        "attempt_count": 0,
        "stop_reason": stop_reason,
        "task_file_status": task_file_status,
    }


def _task_process_entrypoint(
    config_payload: dict[str, Any],
    entry: dict[str, Any],
    result_queue: Any,
) -> None:
    try:
        config = AgentConfig(**config_payload)
        agent = GaiaLangGraphAgent(config=config)
        result = agent.run_task(
            question=str(entry.get("question", "")),
            task_id=str(entry.get("task_id", "")),
            run_label="submission",
            level=str(entry.get("level", "")),
            file_name=str(entry.get("file_name", "")),
        )
        result_queue.put({"ok": True, "result": result})
    except Exception as err:  # noqa: BLE001
        result_queue.put({"ok": False, "error": f"{err.__class__.__name__}: {err}"})


def _run_task_with_hard_timeout(
    config: AgentConfig,
    entry: dict[str, Any],
    task_timeout_seconds: int,
) -> dict[str, Any]:
    if int(task_timeout_seconds) <= 0:
        return _run_task_with_thread_agent(config=config, entry=entry)

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    started = time.perf_counter()

    process = context.Process(
        target=_task_process_entrypoint,
        args=(asdict(config), entry, result_queue),
    )
    process.start()
    process.join(int(task_timeout_seconds))

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)

        elapsed = time.perf_counter() - started
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:  # noqa: BLE001
            pass
        return _build_error_run_result(
            stop_reason=f"task_timeout_{int(task_timeout_seconds)}s",
            latency_seconds=elapsed,
            task_file_status="timeout",
        )

    payload: dict[str, Any] = {}
    try:
        queue_result = result_queue.get(timeout=2)
        if isinstance(queue_result, dict):
            payload = queue_result
    except Exception:  # noqa: BLE001
        payload = {}
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.perf_counter() - started
    if payload.get("ok") and isinstance(payload.get("result"), dict):
        return payload["result"]

    error_message = str(payload.get("error", "missing_result_from_task_process")).strip()
    return _build_error_run_result(
        stop_reason=f"task_process_error: {error_message}",
        latency_seconds=elapsed,
    )


def _ordered_entries(questions: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(questions):
        task_id = str(item.get("task_id", "")).strip()
        question = str(item.get("question", "")).strip()
        level = str(item.get("Level", item.get("level", ""))).strip()
        file_name = str(item.get("file_name", "")).strip()
        if not task_id or not question:
            continue
        entries.append(
            {
                "index": idx,
                "task_id": task_id,
                "question": question,
                "level": level,
                "file_name": file_name,
            }
        )

    if limit > 0:
        return entries[: max(1, int(limit))]
    return entries


def _answers_by_task_from_cache(cache_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for item in cache_payload.get("answers", []) or []:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()
        submitted_answer = str(item.get("submitted_answer", "")).strip()
        if not task_id:
            continue
        mapping[task_id] = {"task_id": task_id, "submitted_answer": submitted_answer}
    return mapping


def _logs_by_task_from_cache(cache_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in cache_payload.get("results_log", []) or []:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("Task ID", "")).strip()
        if not task_id:
            continue
        mapping[task_id] = item
    return mapping


def _historical_usable_answers_for_user(username: str) -> dict[str, dict[str, Any]]:
    safe_user = _safe_username(username)
    cache_files = sorted(SUBMISSION_CACHE_DIR.glob(f"submission_cache_{safe_user}_*.json"))

    history: dict[str, dict[str, Any]] = {}
    for cache_path in cache_files:
        try:
            payload = _load_submission_cache(cache_path)
        except Exception:
            continue

        answer_map = _answers_by_task_from_cache(payload)
        log_map = _logs_by_task_from_cache(payload)

        for task_id, answer_item in answer_map.items():
            submitted_answer = str(answer_item.get("submitted_answer", "")).strip()
            if not submitted_answer or is_unusable_answer(submitted_answer):
                continue

            log_item = log_map.get(task_id, {})
            if not isinstance(log_item, dict):
                log_item = {}

            stop_reason = str(log_item.get("Stop Reason", "")).strip().lower()
            status = str(log_item.get("Status", "")).strip().lower()

            if status and status not in {"success", "resumed", "resumed_history"}:
                continue
            if "timeout" in stop_reason or stop_reason.startswith("task_process_error"):
                continue

            history[task_id] = {
                "submitted_answer": submitted_answer,
                "log": log_item,
                "cache_path": str(cache_path),
            }

    return history


def _cache_payload_from_state(
    username: str,
    worker_count: int,
    entries: list[dict[str, Any]],
    answers_by_index: dict[int, dict[str, str]],
    logs_by_index: dict[int, dict[str, Any]],
    started_at: str,
    resumed_from: str,
    status: str,
) -> dict[str, Any]:
    ordered_indexes = sorted(answers_by_index.keys())
    answers_payload = [answers_by_index[idx] for idx in ordered_indexes]
    results_log = [logs_by_index[idx] for idx in ordered_indexes if idx in logs_by_index]

    return {
        "username": username,
        "agent_code": _get_agent_code_url(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat() if status == "complete" else "",
        "worker_count": worker_count,
        "question_count": len(entries),
        "answers": answers_payload,
        "results_log": results_log,
        "status": status,
        "resumed_from_cache": resumed_from,
    }


def _generate_answers_concurrently(
    config: AgentConfig,
    username: str,
    questions: list[dict[str, Any]],
    worker_count: int,
    resume: bool,
    resume_cache_path: Path | None,
    rerun_unusable_resume_answers: bool,
    use_history_fallback: bool,
    task_timeout_seconds: int,
    save_every: int,
    limit: int,
) -> tuple[dict[str, Any], Path]:
    entries = _ordered_entries(questions=questions, limit=limit)
    if not entries:
        raise ValueError("No valid questions found to process.")

    started_at = datetime.now(timezone.utc).isoformat()
    answers_by_index: dict[int, dict[str, str]] = {}
    logs_by_index: dict[int, dict[str, Any]] = {}

    resumed_from = ""
    resumed_answer_map: dict[str, dict[str, str]] = {}
    resumed_log_map: dict[str, dict[str, Any]] = {}
    history_answer_map: dict[str, dict[str, Any]] = {}
    resumed_from_cache_count = 0
    restored_from_history_count = 0

    if resume:
        source_cache = resume_cache_path
        if source_cache is None:
            source_cache = _latest_cache_for_user(username)
        if source_cache is not None and source_cache.exists():
            payload = _load_submission_cache(source_cache)
            resumed_answer_map = _answers_by_task_from_cache(payload)
            resumed_log_map = _logs_by_task_from_cache(payload)
            resumed_from = str(source_cache)

    if use_history_fallback:
        history_answer_map = _historical_usable_answers_for_user(username=username)

    pending_entries: list[dict[str, Any]] = []
    for entry in entries:
        idx = int(entry["index"])
        task_id = str(entry["task_id"])
        question = str(entry["question"])
        level = str(entry["level"])

        cached_answer = resumed_answer_map.get(task_id)
        if cached_answer is None:
            historical_answer = history_answer_map.get(task_id)
            if historical_answer is not None:
                normalized_answer = normalize_answer(
                    historical_answer.get("submitted_answer", "I don't know"),
                    question=question,
                )
                if not is_unusable_answer(normalized_answer):
                    answers_by_index[idx] = {"task_id": task_id, "submitted_answer": normalized_answer}
                    historical_log = historical_answer.get("log", {})
                    if not isinstance(historical_log, dict):
                        historical_log = {}
                    logs_by_index[idx] = {
                        "Task ID": task_id,
                        "Submitted Answer": normalized_answer,
                        "Level": level,
                        "Latency (s)": historical_log.get("Latency (s)", ""),
                        "Attempts": historical_log.get("Attempts", ""),
                        "Stop Reason": historical_log.get("Stop Reason", "history_fallback"),
                        "Task File Status": historical_log.get("Task File Status", ""),
                        "Status": "resumed_history",
                        "Source Cache": historical_answer.get("cache_path", ""),
                    }
                    restored_from_history_count += 1
                    continue
            pending_entries.append(entry)
            continue

        normalized_answer = normalize_answer(cached_answer.get("submitted_answer", "I don't know"), question=question)
        if rerun_unusable_resume_answers and is_unusable_answer(normalized_answer):
            historical_answer = history_answer_map.get(task_id)
            if historical_answer is not None:
                restored_answer = normalize_answer(
                    historical_answer.get("submitted_answer", "I don't know"),
                    question=question,
                )
                if not is_unusable_answer(restored_answer):
                    answers_by_index[idx] = {"task_id": task_id, "submitted_answer": restored_answer}
                    historical_log = historical_answer.get("log", {})
                    if not isinstance(historical_log, dict):
                        historical_log = {}
                    logs_by_index[idx] = {
                        "Task ID": task_id,
                        "Submitted Answer": restored_answer,
                        "Level": level,
                        "Latency (s)": historical_log.get("Latency (s)", ""),
                        "Attempts": historical_log.get("Attempts", ""),
                        "Stop Reason": historical_log.get("Stop Reason", "history_fallback"),
                        "Task File Status": historical_log.get("Task File Status", ""),
                        "Status": "resumed_history",
                        "Source Cache": historical_answer.get("cache_path", ""),
                    }
                    restored_from_history_count += 1
                    continue
            pending_entries.append(entry)
            continue

        answers_by_index[idx] = {"task_id": task_id, "submitted_answer": normalized_answer}
        resumed_from_cache_count += 1

        cached_log = resumed_log_map.get(task_id, {})
        logs_by_index[idx] = {
            "Task ID": task_id,
            "Submitted Answer": normalized_answer,
            "Level": level,
            "Latency (s)": cached_log.get("Latency (s)", ""),
            "Attempts": cached_log.get("Attempts", ""),
            "Stop Reason": cached_log.get("Stop Reason", "resumed_from_cache"),
            "Task File Status": cached_log.get("Task File Status", ""),
            "Status": cached_log.get("Status", "resumed"),
        }

    cache_path = _cache_path_for_user(username)
    in_progress_payload = _cache_payload_from_state(
        username=username,
        worker_count=worker_count,
        entries=entries,
        answers_by_index=answers_by_index,
        logs_by_index=logs_by_index,
        started_at=started_at,
        resumed_from=resumed_from,
        status="running",
    )
    _write_submission_cache(cache_path, in_progress_payload)

    total_count = len(entries)
    resumed_count = len(answers_by_index)
    print(
        f"Prepared {total_count} tasks for user '{username}'. "
        f"Resumed {resumed_from_cache_count} from cache; restored {restored_from_history_count} from history; "
        f"pending {len(pending_entries)}."
    )
    print(
        f"Using {worker_count} workers. Per-task hard timeout: {task_timeout_seconds}s. "
        f"Cache path: {cache_path}"
    )

    save_every = max(1, int(save_every))
    started_perf = time.perf_counter()

    completed_since_resume = 0
    if worker_count == 1:
        for entry in pending_entries:
            idx = int(entry["index"])
            task_id = str(entry["task_id"])
            question = str(entry["question"])
            level = str(entry["level"])

            try:
                run_result = _run_task_with_hard_timeout(
                    config=config,
                    entry=entry,
                    task_timeout_seconds=task_timeout_seconds,
                )
            except Exception as err:  # noqa: BLE001
                run_result = {
                    "status": "error",
                    "submitted_answer": "I don't know",
                    "latency_seconds": "",
                    "attempt_count": 0,
                    "stop_reason": f"thread_exception: {err}",
                    "task_file_status": "error",
                }

            submitted_answer = normalize_answer(run_result.get("submitted_answer", "I don't know"), question=question)
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

            completed_since_resume += 1
            total_completed = len(answers_by_index)
            elapsed = time.perf_counter() - started_perf
            avg_new = elapsed / completed_since_resume if completed_since_resume else 0.0
            remaining_new = len(pending_entries) - completed_since_resume
            eta = avg_new * remaining_new
            print(
                f"[{total_completed}/{total_count}] {task_id} done | "
                f"status={run_result.get('status', '')} | elapsed={elapsed:.1f}s | eta~{eta:.1f}s"
            )

            if completed_since_resume % save_every == 0 or total_completed == total_count:
                running_payload = _cache_payload_from_state(
                    username=username,
                    worker_count=worker_count,
                    entries=entries,
                    answers_by_index=answers_by_index,
                    logs_by_index=logs_by_index,
                    started_at=started_at,
                    resumed_from=resumed_from,
                    status="running",
                )
                _write_submission_cache(cache_path, running_payload)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _run_task_with_hard_timeout,
                    config,
                    entry,
                    task_timeout_seconds,
                ): entry
                for entry in pending_entries
            }

            for future in as_completed(future_map):
                entry = future_map[future]
                idx = int(entry["index"])
                task_id = str(entry["task_id"])
                question = str(entry["question"])
                level = str(entry["level"])

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

                submitted_answer = normalize_answer(run_result.get("submitted_answer", "I don't know"), question=question)
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

                completed_since_resume += 1
                total_completed = len(answers_by_index)
                elapsed = time.perf_counter() - started_perf
                avg_new = elapsed / completed_since_resume if completed_since_resume else 0.0
                remaining_new = len(pending_entries) - completed_since_resume
                eta = avg_new * remaining_new
                print(
                    f"[{total_completed}/{total_count}] {task_id} done | "
                    f"status={run_result.get('status', '')} | elapsed={elapsed:.1f}s | eta~{eta:.1f}s"
                )

                if completed_since_resume % save_every == 0 or total_completed == total_count:
                    running_payload = _cache_payload_from_state(
                        username=username,
                        worker_count=worker_count,
                        entries=entries,
                        answers_by_index=answers_by_index,
                        logs_by_index=logs_by_index,
                        started_at=started_at,
                        resumed_from=resumed_from,
                        status="running",
                    )
                    _write_submission_cache(cache_path, running_payload)

    complete_payload = _cache_payload_from_state(
        username=username,
        worker_count=worker_count,
        entries=entries,
        answers_by_index=answers_by_index,
        logs_by_index=logs_by_index,
        started_at=started_at,
        resumed_from=resumed_from,
        status="complete",
    )
    _write_submission_cache(cache_path, complete_payload)
    return complete_payload, cache_path


def _submit_cached_payload(config: AgentConfig, username: str, cache_payload: dict[str, Any]) -> dict[str, Any]:
    answers_payload = cache_payload.get("answers", [])
    if not isinstance(answers_payload, list) or not answers_payload:
        raise ValueError("Cache payload contains no answers.")

    submission_data = {
        "username": username,
        "agent_code": _get_agent_code_url(),
        "answers": answers_payload,
    }
    return _submit_answers(config.api_url, submission_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concurrent GAIA fast-submit runner that mirrors Gradio cache/submit flow "
            "under your Hugging Face username."
        )
    )
    parser.add_argument(
        "--username",
        default="",
        help="Hugging Face username. If omitted, resolved from HF token via whoami.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Number of concurrent workers (1-20).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of questions (0 means all).",
    )
    parser.add_argument(
        "--cache-path",
        default="",
        help=(
            "Optional cache file path. For --submit-only, this path is submitted. "
            "For generation, this acts as a resume source."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist running cache every N completed tasks.",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=_default_task_timeout_seconds(),
        help=(
            "Hard timeout per task in seconds (0 disables hard timeout). "
            "Default: 1200 or GAIA_FAST_TASK_TIMEOUT_SECONDS."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from previous cache.",
    )
    parser.add_argument(
        "--keep-idk-resume",
        action="store_true",
        help="Keep resumed 'I don't know' answers instead of rerunning them.",
    )
    parser.add_argument(
        "--no-history-fallback",
        action="store_true",
        help="Disable restoring usable answers from older caches for the same task_id.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--cache-only",
        action="store_true",
        help="Generate and cache answers only (do not submit).",
    )
    mode.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit a previously generated cache without generating new answers.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    worker_count = max(1, min(int(args.workers), 20))
    config = AgentConfig.from_env()
    username = _resolve_username(args.username)

    explicit_cache_path = Path(args.cache_path).expanduser() if str(args.cache_path).strip() else None

    print(f"User: {username}")
    print(f"API URL: {config.api_url}")

    if args.submit_only:
        target_cache = explicit_cache_path or _latest_cache_for_user(username)
        if target_cache is None:
            raise FileNotFoundError(
                "No cache found for user. Run generation first or pass --cache-path for submit-only mode."
            )
        cache_payload = _load_submission_cache(target_cache)
        result = _submit_cached_payload(config=config, username=username, cache_payload=cache_payload)
        print("Submission successful")
        print(
            f"Score: {result.get('score', 'N/A')}% "
            f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')})"
        )
        print(f"Message: {result.get('message', 'No message')}")
        print(f"Cache submitted: {target_cache}")
        return

    questions = _fetch_questions(config.api_url)
    cache_payload, cache_path = _generate_answers_concurrently(
        config=config,
        username=username,
        questions=questions,
        worker_count=worker_count,
        resume=not args.no_resume,
        resume_cache_path=explicit_cache_path,
        rerun_unusable_resume_answers=not args.keep_idk_resume,
        use_history_fallback=not args.no_history_fallback,
        task_timeout_seconds=max(0, int(args.task_timeout_seconds)),
        save_every=max(1, int(args.save_every)),
        limit=max(0, int(args.limit)),
    )

    answer_count = len(cache_payload.get("answers", []) or [])
    print(f"Generation complete. Answers cached: {answer_count}")
    print(f"Cache file: {cache_path}")

    if args.cache_only:
        return

    result = _submit_cached_payload(config=config, username=username, cache_payload=cache_payload)
    print("Submission successful")
    print(
        f"Score: {result.get('score', 'N/A')}% "
        f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')})"
    )
    print(f"Message: {result.get('message', 'No message')}")
    print(f"Cache submitted: {cache_path}")


if __name__ == "__main__":
    main()
