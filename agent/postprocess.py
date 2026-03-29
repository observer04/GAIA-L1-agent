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
_UNCERTAIN_FRAGMENTS = (
    "cannot determine",
    "can't determine",
    "unable to determine",
    "insufficient information",
    "not enough information",
)


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

    compact = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    if compact in {"i don t know", "i do not know", "idk", "not sure", "unknown", "n a", "none"}:
        return True
    if any(fragment in lowered for fragment in _UNCERTAIN_FRAGMENTS):
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

    if len(text) >= 2:
        opening, closing = text[0], text[-1]
        if (opening, closing) in {("(", ")"), ("[", "]"), ("{", "}")}:
            inner = text[1:-1].strip()
            if inner:
                text = inner

    text = re.sub(r"\s+", " ", text).strip()

    # Canonicalize plain numeric answers written with thousands separators/trailing zeros.
    compact_numeric_source = re.sub(r"\s+", "", text)
    numeric_candidate = compact_numeric_source.replace(",", "")
    if re.fullmatch(r"[-+0-9,\.\s]+", text) and re.fullmatch(r"[-+]?\d+(\.\d+)?", numeric_candidate):
        if "." in numeric_candidate:
            numeric_candidate = numeric_candidate.rstrip("0").rstrip(".")
        text = numeric_candidate
    else:
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r",\s*,+", ", ", text)
        text = text.strip()

    return text or "I don't know"


def normalize_for_submission(raw_output: str) -> str:
    return normalize_answer(raw_output)
