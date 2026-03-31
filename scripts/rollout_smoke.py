from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import requests


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str


def _normalize_base_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise ValueError("base URL is empty")
    return value.rstrip("/")


def _check_get_json(url: str, timeout_seconds: int, expected_status: int = 200) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException as err:
        return False, f"request failed: {err}"

    elapsed = time.perf_counter() - start
    if response.status_code != expected_status:
        return False, f"status {response.status_code} (expected {expected_status})"

    try:
        payload = response.json()
    except ValueError:
        content_type = response.headers.get("content-type", "")
        preview = response.text[:120].replace("\n", " ")
        return False, (
            f"response is not valid JSON (status={response.status_code}, "
            f"content-type={content_type!r}, body_preview={preview!r})"
        )

    return True, f"status={response.status_code} elapsed={elapsed:.2f}s keys={list(payload.keys())[:5]}"


def _check_chat_stream(base_url: str, timeout_seconds: int, ttfb_budget_seconds: float) -> tuple[bool, str]:
    url = f"{base_url}/api/chat/stream"
    payload = {
        "message": "Reply with exactly: pong",
        "run_label": "rollout-smoke",
    }

    start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds, stream=True)
    except requests.RequestException as err:
        return False, f"stream request failed: {err}"

    if response.status_code != 200:
        body_preview = response.text[:300]
        return False, f"status {response.status_code}; body={body_preview!r}"

    first_event_name = ""
    first_event_data = ""

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        line = raw_line.strip()
        if not line:
            if first_event_name or first_event_data:
                break
            continue

        if line.startswith("event:") and not first_event_name:
            first_event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and not first_event_data:
            first_event_data = line.split(":", 1)[1].strip()

    elapsed = time.perf_counter() - start
    if not first_event_name and not first_event_data:
        return False, "no SSE event/data received"

    if elapsed > ttfb_budget_seconds:
        return False, f"TTFB {elapsed:.2f}s exceeded budget {ttfb_budget_seconds:.2f}s"

    detail = f"first_event={first_event_name or 'unknown'} ttfb={elapsed:.2f}s"
    if first_event_data:
        try:
            parsed = json.loads(first_event_data)
            run_id = str(parsed.get("run_id", ""))
            if run_id:
                detail += f" run_id={run_id[:16]}"
        except json.JSONDecodeError:
            pass

    return True, detail


def run_smoke(
    *,
    base_url: str,
    timeout_seconds: int,
    ttfb_budget_seconds: float,
    skip_chat: bool,
) -> list[SmokeResult]:
    results: list[SmokeResult] = []

    ok, detail = _check_get_json(f"{base_url}/api/health", timeout_seconds=timeout_seconds)
    results.append(SmokeResult(name="health", ok=ok, detail=detail))

    ok, detail = _check_get_json(f"{base_url}/api", timeout_seconds=timeout_seconds)
    results.append(SmokeResult(name="api_root", ok=ok, detail=detail))

    if skip_chat:
        results.append(SmokeResult(name="chat_stream", ok=True, detail="skipped"))
    else:
        ok, detail = _check_chat_stream(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            ttfb_budget_seconds=ttfb_budget_seconds,
        )
        results.append(SmokeResult(name="chat_stream", ok=ok, detail=detail))

    return results


def _print_summary(results: list[SmokeResult], base_url: str) -> None:
    print(f"Rollout smoke summary for {base_url}")
    print("-" * 72)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rollout smoke checks for /gaia_agent deployment")
    parser.add_argument(
        "--base-url",
        default=os.getenv("GAIA_DEPLOY_BASE_URL", "http://localhost:8000/gaia_agent"),
        help="Base app URL including path prefix, e.g. https://ommprakash.cloud/gaia_agent",
    )
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--ttfb-budget-seconds", type=float, default=20.0)
    parser.add_argument("--skip-chat", action="store_true", help="Skip streaming chat endpoint check")
    args = parser.parse_args()

    base_url = _normalize_base_url(args.base_url)
    results = run_smoke(
        base_url=base_url,
        timeout_seconds=max(5, args.timeout_seconds),
        ttfb_budget_seconds=max(1.0, args.ttfb_budget_seconds),
        skip_chat=args.skip_chat,
    )

    _print_summary(results, base_url)

    failed = [item for item in results if not item.ok]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
