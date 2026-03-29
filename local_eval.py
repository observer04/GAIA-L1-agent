from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from agent import GaiaLangGraphAgent
from agent.config import AgentConfig


def fetch_questions(api_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{api_url.rstrip('/')}/questions", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Questions response is not a list")
    return payload


def fetch_random_question(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url.rstrip('/')}/random-question", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Random question response is not an object")
    return payload


def evaluate_questions(
    agent: GaiaLangGraphAgent,
    questions: list[dict[str, Any]],
    run_label: str,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    for item in questions:
        task_id = str(item.get("task_id", "")).strip()
        question = str(item.get("question", "")).strip()
        level = str(item.get("Level", item.get("level", ""))).strip()
        file_name = str(item.get("file_name", "")).strip()
        if not task_id or not question:
            continue

        run_result = agent.run_task(
            question=question,
            task_id=task_id,
            run_label=run_label,
            level=level,
            file_name=file_name,
        )
        run_result["question"] = question
        run_result["level"] = level
        run_result["file_name"] = file_name
        results.append(run_result)

    success_count = sum(1 for r in results if r.get("status") == "success")
    avg_latency = (
        round(sum(float(r.get("latency_seconds", 0.0)) for r in results) / len(results), 3)
        if results
        else 0.0
    )
    stop_reason_counts: dict[str, int] = {}
    tool_error_counts: dict[str, int] = {}
    idk_answers = 0

    for result in results:
        stop_reason = str(result.get("stop_reason", "") or "unknown").strip() or "unknown"
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1

        submitted_answer = str(result.get("submitted_answer", "")).strip().lower()
        if submitted_answer in {"i don't know", "idk", "unknown", "not sure"}:
            idk_answers += 1

        for item in result.get("tool_trace", []) or []:
            tool_name = str(item.get("tool", "unknown")).strip() or "unknown"
            preview = str(item.get("preview", "")).strip()
            if preview.upper().startswith("ERROR:"):
                first_line = preview.splitlines()[0][:120]
                key = f"{tool_name}: {first_line}"
                tool_error_counts[key] = tool_error_counts.get(key, 0) + 1

    return {
        "run_label": run_label,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "successful_runs": success_count,
        "avg_latency_seconds": avg_latency,
        "idk_answers": idk_answers,
        "stop_reason_counts": stop_reason_counts,
        "tool_error_counts": tool_error_counts,
        "results": results,
    }


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"local_eval_{report['run_label']}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _dedupe_task_ids(task_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in task_ids:
        task_id = str(item or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        ordered.append(task_id)
    return ordered


def _load_manual_tasks(tasks_json: Path) -> list[dict[str, Any]]:
    if not tasks_json.exists():
        raise FileNotFoundError(f"Tasks JSON not found: {tasks_json}")

    payload = json.loads(tasks_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        raw_tasks = payload.get("tasks", [])
    elif isinstance(payload, list):
        raw_tasks = payload
    else:
        raise ValueError("Tasks JSON must be a list or an object with a 'tasks' list")

    tasks: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise ValueError(f"Task entry at index {idx} is not an object")

        task_id = str(item.get("task_id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not task_id or not question:
            raise ValueError(f"Task entry at index {idx} missing task_id/question")

        tasks.append(
            {
                "task_id": task_id,
                "question": question,
                "Level": item.get("Level", item.get("level", "")),
                "file_name": item.get("file_name", ""),
            }
        )

    return tasks


def _select_questions_by_task_ids(
    api_url: str,
    requested_task_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    all_questions = fetch_questions(api_url)
    by_task_id: dict[str, dict[str, Any]] = {}
    for item in all_questions:
        task_id = str(item.get("task_id", "")).strip()
        if task_id:
            by_task_id[task_id] = item

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in requested_task_ids:
        matched = by_task_id.get(task_id)
        if matched is None:
            missing.append(task_id)
            continue
        selected.append(matched)

    return selected, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GAIA evaluator for LangGraph v2 agent")
    parser.add_argument(
        "--mode",
        choices=["random", "sample", "full", "focused"],
        default="random",
        help="Evaluation mode",
    )
    parser.add_argument("--sample-size", type=int, default=5, help="Sample size for sample mode")
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Task ID to evaluate (repeat flag for multiple task IDs) when mode=focused",
    )
    parser.add_argument(
        "--tasks-json",
        default="",
        help="Path to JSON list/object of manual tasks when mode=focused",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional custom run label",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail focused mode if any requested task IDs are missing from /questions",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/local_eval",
        help="Directory for local evaluation reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig.from_env()
    agent = GaiaLangGraphAgent(config=config)

    if args.mode == "random":
        questions = [fetch_random_question(config.api_url)]
        run_label = "random"
        missing_task_ids: list[str] = []
        requested_task_ids: list[str] = [str(questions[0].get("task_id", "")).strip()]
    elif args.mode == "focused":
        tasks_json = str(args.tasks_json or "").strip()
        if tasks_json:
            questions = _load_manual_tasks(Path(tasks_json))
            requested_task_ids = [str(item.get("task_id", "")).strip() for item in questions]
            missing_task_ids = []
            run_label = args.run_label.strip() or f"focused_manual_{len(questions)}"
        else:
            requested_task_ids = _dedupe_task_ids(list(args.task_id or []))
            if not requested_task_ids:
                raise ValueError("Focused mode requires --task-id (repeatable) or --tasks-json")

            questions, missing_task_ids = _select_questions_by_task_ids(
                api_url=config.api_url,
                requested_task_ids=requested_task_ids,
            )
            if args.strict_missing and missing_task_ids:
                raise ValueError(f"Missing task IDs from /questions: {missing_task_ids}")
            if not questions:
                raise ValueError("No matching questions found for provided --task-id values")
            run_label = args.run_label.strip() or f"focused_{len(questions)}"
    else:
        all_questions = fetch_questions(config.api_url)
        if args.mode == "sample":
            questions = all_questions[: max(1, int(args.sample_size))]
            run_label = f"sample_{len(questions)}"
            missing_task_ids = []
            requested_task_ids = [str(item.get("task_id", "")).strip() for item in questions]
        else:
            questions = all_questions
            run_label = "full"
            missing_task_ids = []
            requested_task_ids = [str(item.get("task_id", "")).strip() for item in questions]

    report = evaluate_questions(agent=agent, questions=questions, run_label=run_label)
    report["mode"] = args.mode
    report["requested_task_ids"] = requested_task_ids
    report["evaluated_task_ids"] = [str(item.get("task_id", "")).strip() for item in questions]
    report["missing_task_ids"] = missing_task_ids
    output_path = save_report(report=report, output_dir=Path(args.output_dir))

    print(f"Saved local evaluation report to: {output_path}")
    print(
        f"Questions: {report['total_questions']} | "
        f"Successful runs: {report['successful_runs']} | "
        f"Avg latency: {report['avg_latency_seconds']}s"
    )


if __name__ == "__main__":
    main()
