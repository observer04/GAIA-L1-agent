import os

import gradio as gr
import pandas as pd
import requests

from agent import GaiaLangGraphAgent, normalize_answer
from agent.config import AgentConfig


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


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetch questions, run the GAIA LangGraph v2 agent, submit answers, and return logs."""
    if not profile:
        return "Please login to Hugging Face first.", None

    username = profile.username
    config = AgentConfig.from_env()
    api_url = config.api_url

    try:
        agent = GaiaLangGraphAgent(config=config)
    except Exception as err:  # noqa: BLE001
        return f"Error initializing agent: {err}", None

    space_id = os.getenv("SPACE_ID", "")
    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "https://huggingface.co/spaces/<your-space>/tree/main"
    )

    try:
        questions = _fetch_questions(api_url)
    except Exception as err:  # noqa: BLE001
        return f"Error fetching questions: {err}", None

    results_log: list[dict] = []
    answers_payload: list[dict] = []

    for item in questions:
        task_id = item.get("task_id", "")
        question = item.get("question", "")
        if not task_id or not question:
            continue

        run_result = agent.run_task(question=question, task_id=task_id, run_label="submission")
        submitted_answer = normalize_answer(run_result.get("submitted_answer", "I don't know"))

        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append(
            {
                "Task ID": task_id,
                "Submitted Answer": submitted_answer,
                "Latency (s)": run_result.get("latency_seconds", ""),
                "Attempts": run_result.get("attempt_count", ""),
                "Stop Reason": run_result.get("stop_reason", ""),
                "Status": run_result.get("status", ""),
            }
        )

    if not answers_payload:
        return "Agent did not generate any answers.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }

    try:
        result_data = _submit_answers(api_url, submission_data)
        status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as err:
        details = f"HTTP {err.response.status_code}"
        try:
            payload = err.response.json()
            details += f" - {payload.get('detail', '')}"
        except Exception:  # noqa: BLE001
            details += f" - {err.response.text[:500]}"
        return f"Submission Failed: {details}", pd.DataFrame(results_log)
    except Exception as err:  # noqa: BLE001
        return f"Submission Failed: {err}", pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner (LangGraph v2)")
    gr.Markdown(
        """
        1. Login with your Hugging Face account.
        2. Click **Run Evaluation & Submit All Answers**.
        3. Review the per-task logs and final score.

        Notes:
        - The benchmark checks exact-match outputs.
        - This app submits only normalized answer strings (no reasoning prefixes).
        """
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=6, interactive=False)
    results_table = gr.DataFrame(label="Per-task Run Logs", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
