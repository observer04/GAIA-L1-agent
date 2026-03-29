from __future__ import annotations

import ast
import contextlib
import io
import math
import subprocess
import statistics
import traceback
from typing import Any

from .runtime import get_runtime_working_dir

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


class StatefulPythonExecutor:
    """Minimal stateful Python executor for multi-turn agent traces."""

    def __init__(self) -> None:
        globals_map: dict[str, Any] = {
            "__builtins__": __builtins__,
            "math": math,
            "statistics": statistics,
        }
        if pd is not None:
            globals_map["pd"] = pd
        if np is not None:
            globals_map["np"] = np

        self._globals = globals_map

    @staticmethod
    def _strip_fences(code: str) -> str:
        cleaned = (code or "").strip()
        if cleaned.startswith("```python"):
            cleaned = cleaned[len("```python") :]
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```") :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        return cleaned.strip()

    def run(self, code: str) -> str:
        cleaned = self._strip_fences(code)
        if not cleaned:
            return "ERROR: empty python code"

        stdout_buffer = io.StringIO()
        try:
            tree = ast.parse(cleaned)
            tail_expression = None
            body_to_exec = cleaned

            if tree.body and isinstance(tree.body[-1], ast.Expr):
                tail_expression = ast.get_source_segment(cleaned, tree.body[-1].value)
                prefix_lines = cleaned.splitlines()
                # Remove the final expression line from executed body to avoid duplicate output.
                body_to_exec = "\n".join(prefix_lines[:-1]).strip()

            with contextlib.redirect_stdout(stdout_buffer):
                if body_to_exec:
                    exec(
                        compile(body_to_exec, "<gaia-agent-python>", "exec"),
                        self._globals,
                        self._globals,
                    )

                tail_value = None
                if tail_expression:
                    tail_value = eval(
                        compile(tail_expression, "<gaia-agent-python-tail>", "eval"),
                        self._globals,
                        self._globals,
                    )

            stdout = stdout_buffer.getvalue().strip()
            if stdout and tail_expression:
                return f"{stdout}\n{tail_value}".strip()
            if stdout:
                return stdout
            if tail_expression:
                return str(tail_value)

            for result_name in ("final_answer", "answer", "result"):
                if result_name in self._globals:
                    return str(self._globals[result_name])

            return "OK: python code executed without stdout"
        except Exception as err:  # noqa: BLE001
            trace = traceback.format_exc(limit=2)
            return (
                "ERROR: python execution failed "
                f"({err.__class__.__name__}: {err})\n{trace}"
            )


class BashCommandExecutor:
    """Minimal shell executor with timeout and captured output."""

    def __init__(self, timeout_seconds: int = 15) -> None:
        self._timeout_seconds = max(1, int(timeout_seconds))

    @staticmethod
    def _strip_fences(command: str) -> str:
        cleaned = (command or "").strip()
        if cleaned.startswith("```bash"):
            cleaned = cleaned[len("```bash") :]
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```") :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        return cleaned.strip()

    def run(self, command: str) -> str:
        cleaned = self._strip_fences(command)
        if not cleaned:
            return "ERROR: empty bash command"

        runtime_dir = get_runtime_working_dir().strip()
        cwd = runtime_dir or None

        try:
            completed = subprocess.run(
                cleaned,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as err:
            partial = (err.stdout or "").strip()
            if err.stderr:
                partial = f"{partial}\n{err.stderr.strip()}".strip()
            return (
                f"ERROR: bash command timed out after {self._timeout_seconds}s"
                + (f"\n{partial}" if partial else "")
            )
        except Exception as err:  # noqa: BLE001
            return f"ERROR: bash execution failed ({err.__class__.__name__}: {err})"

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        combined = f"{stdout}\n{stderr}".strip() if stderr else stdout

        if completed.returncode != 0:
            prefix = f"ERROR: bash command failed (exit {completed.returncode})"
            return f"{prefix}\n{combined}".strip() if combined else prefix

        return combined or "OK: bash command executed without output"
