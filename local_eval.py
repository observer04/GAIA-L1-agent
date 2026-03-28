from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
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
    started_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []

    for item in questions:
        task_id = str(item.get("task_id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not task_id or not question:
            continue

        run_result = agent.run_task(question=question, task_id=task_id, run_label=run_label)
        run_result["question"] = question
        results.append(run_result)

    success_count = sum(1 for r in results if r.get("status") == "success")
    avg_latency = (
        round(sum(float(r.get("latency_seconds", 0.0)) for r in results) / len(results), 3)
        if results
        else 0.0
    )

    return {
        "run_label": run_label,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "total_questions": len(results),
        "successful_runs": success_count,
        "avg_latency_seconds": avg_latency,
        "results": results,
    }


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"local_eval_{report['run_label']}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GAIA evaluator for LangGraph v2 agent")
    parser.add_argument(
        "--mode",
        choices=["random", "sample", "full"],
        default="random",
        help="Evaluation mode",
    )
    parser.add_argument("--sample-size", type=int, default=5, help="Sample size for sample mode")
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
    else:
        all_questions = fetch_questions(config.api_url)
        if args.mode == "sample":
            questions = all_questions[: max(1, int(args.sample_size))]
            run_label = f"sample_{len(questions)}"
        else:
            questions = all_questions
            run_label = "full"

    report = evaluate_questions(agent=agent, questions=questions, run_label=run_label)
    output_path = save_report(report=report, output_dir=Path(args.output_dir))

    print(f"Saved local evaluation report to: {output_path}")
    print(
        f"Questions: {report['total_questions']} | "
        f"Successful runs: {report['successful_runs']} | "
        f"Avg latency: {report['avg_latency_seconds']}s"
    )


if __name__ == "__main__":
    main()
