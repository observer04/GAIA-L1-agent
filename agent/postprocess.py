from __future__ import annotations

import re


_FINAL_ANSWER_PATTERN = re.compile(
    r"FINAL\s*ANSWER\s*:\s*(.+?)(?:\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_PREFIX_PATTERN = re.compile(
    r"^\s*(?:FINAL\s*ANSWER|Answer|Result|Response)\s*:\s*",
    flags=re.IGNORECASE,
)
_ERROR_LIKE_PATTERN = re.compile(r"^\s*(?:ERROR|EXCEPTION)\b", flags=re.IGNORECASE)
_UNCERTAIN_ANSWERS = {
    "i don't know",
    "idk",
    "unknown",
    "not sure",
    "n/a",
    "none",
}


def has_final_answer_marker(raw_output: str) -> bool:
    text = str(raw_output or "").strip()
    if not text:
        return False
    return _FINAL_ANSWER_PATTERN.search(text) is not None


def is_unusable_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in _UNCERTAIN_ANSWERS:
        return True
    if _ERROR_LIKE_PATTERN.match(text):
        return True
    return False


def normalize_answer(raw_output: str) -> str:
    """Normalize model output into a strict exact-match candidate answer."""
    text = str(raw_output or "").strip()
    if not text:
        return "I don't know"

    matched = _FINAL_ANSWER_PATTERN.search(text)
    if matched:
        text = matched.group(1)

    text = _PREFIX_PATTERN.sub("", text)
    text = text.strip().strip("` ").strip("\"'")

    if "\n" in text:
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if non_empty_lines:
            text = non_empty_lines[0]

    text = re.sub(r"\s+", " ", text).strip()

    # Canonicalize plain numeric answers written with thousands separators.
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")

    return text or "I don't know"


def normalize_for_submission(raw_output: str) -> str:
    return normalize_answer(raw_output)
