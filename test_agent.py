import pytest

from agent.postprocess import has_final_answer_marker, is_unusable_answer, normalize_answer


def test_normalize_answer_with_prefix():
    assert normalize_answer("FINAL ANSWER: 42") == "42"


def test_normalize_answer_plain_text():
    assert normalize_answer("   Paris   ") == "Paris"


def test_normalize_answer_wrapped_result():
    assert normalize_answer("Result:   apple, banana") == "apple, banana"


def test_normalize_answer_removes_thousands_separator_for_plain_number():
    assert normalize_answer("FINAL ANSWER: 1,234") == "1234"


def test_normalize_answer_takes_first_meaningful_line():
    raw = "FINAL ANSWER: Paris\nReasoning: capital city"
    assert normalize_answer(raw) == "Paris"


def test_has_final_answer_marker_detects_final_line():
    assert has_final_answer_marker("Reasoning...\nFINAL ANSWER: 17") is True
    assert has_final_answer_marker("Answer: 17") is False


def test_is_unusable_answer_for_error_and_uncertainty():
    assert is_unusable_answer("ERROR: model invocation failed") is True
    assert is_unusable_answer("I don't know") is True
    assert is_unusable_answer("42") is False


def test_tools_registry_has_required_tools():
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain_community")

    from tools import get_tools

    tool_names = {tool.name for tool in get_tools()}
    assert "download_task_file" in tool_names
    assert "web_search" in tool_names
    assert "fetch_webpage_text" in tool_names
    assert "get_youtube_video_context" in tool_names
    assert "execute_python" in tool_names
