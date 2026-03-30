import pytest
from langchain_core.messages import AIMessage
from typing import cast

from agent.config import AgentConfig
from agent.graph import GaiaLangGraphAgent
from agent.postprocess import has_final_answer_marker, is_unusable_answer, normalize_answer
from agent.state import GaiaAgentState
from agent.tools.core import (
    _is_potential_benchmark_leak_content,
    _is_potential_benchmark_leak_url,
    _merge_and_rank_search_hits,
    _resolve_search_providers,
    _search_google_custom_search_hits,
    _url_dedupe_key,
    download_task_file_raw,
    web_search,
)


def test_normalize_answer_with_prefix():
    assert normalize_answer("FINAL ANSWER: 42") == "42"


def test_normalize_answer_plain_text():
    assert normalize_answer("   Paris   ") == "Paris"


def test_normalize_answer_wrapped_result():
    assert normalize_answer("Result:   apple, banana") == "apple, banana"


def test_normalize_answer_removes_thousands_separator_for_plain_number():
    assert normalize_answer("FINAL ANSWER: 1,234") == "1234"


def test_normalize_answer_normalizes_numeric_trailing_zeros():
    assert normalize_answer("FINAL ANSWER: 1,234.00") == "1234"


def test_normalize_answer_normalizes_list_spacing():
    assert normalize_answer("FINAL ANSWER: apple,banana ,  cherry") == "apple, banana, cherry"


def test_normalize_answer_preserves_numeric_comma_list_when_question_requests_it():
    question = "Please provide just the page numbers as a comma-delimited list in ascending order."
    raw = "FINAL ANSWER: 132,133,134,197,245"
    assert normalize_answer(raw, question=question) == "132, 133, 134, 197, 245"


def test_normalize_answer_takes_first_meaningful_line():
    raw = "FINAL ANSWER: Paris\nReasoning: capital city"
    assert normalize_answer(raw) == "Paris"


def test_has_final_answer_marker_detects_final_line():
    assert has_final_answer_marker("Reasoning...\nFINAL ANSWER: 17") is True
    assert has_final_answer_marker("Answer: 17") is False


def test_is_unusable_answer_for_error_and_uncertainty():
    assert is_unusable_answer("ERROR: model invocation failed") is True
    assert is_unusable_answer("I don't know") is True
    assert is_unusable_answer("I don't know.") is True
    assert is_unusable_answer("Unable to determine from provided evidence") is True
    assert is_unusable_answer("42") is False


def test_tools_registry_has_required_tools():
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain_community")

    from tools import get_tools

    tool_names = {tool.name for tool in get_tools()}
    assert "download_task_file" in tool_names
    assert "web_search" in tool_names
    assert "fetch_webpage_text" in tool_names
    assert "fetch_wikipedia_section" in tool_names
    assert "get_youtube_video_context" in tool_names
    assert "analyze_image_with_vlm" in tool_names
    assert "arxiv_search" in tool_names
    assert "execute_python" in tool_names
    assert "execute_bash" in tool_names
    assert "extract_office_text" in tool_names
    assert "inspect_archive_file" in tool_names
    assert "transcribe_audio_file" in tool_names


def _make_agent_for_evidence_tests(max_iterations: int = 12) -> GaiaLangGraphAgent:
    agent = GaiaLangGraphAgent.__new__(GaiaLangGraphAgent)
    agent.config = AgentConfig(google_api_key="test-key", max_iterations=max_iterations)
    return agent


def test_evidence_check_detects_stagnation_and_adds_hint():
    agent = _make_agent_for_evidence_tests(max_iterations=12)
    state = {
        "question": "dummy question",
        "attempt_count": 5,
        "messages": [AIMessage(content="Working...")],
        "last_error": "",
        "tool_trace": [
            {"tool": "web_search", "preview": "tool_call"},
            {"tool": "web_search", "preview": "tool_call"},
            {"tool": "web_search", "preview": "tool_call"},
            {"tool": "web_search", "preview": "tool_call"},
        ],
    }

    result = agent._evidence_check_node(cast(GaiaAgentState, state))
    assert result.get("stop_reason") == "stagnation_detected"
    hint_messages = result.get("messages", [])
    assert hint_messages
    hint_text = str(getattr(hint_messages[0], "content", ""))
    assert "fetch_webpage_text" in hint_text


def test_evidence_check_uses_synthesis_at_max_iterations(monkeypatch: pytest.MonkeyPatch):
    agent = _make_agent_for_evidence_tests(max_iterations=3)
    monkeypatch.setattr(agent, "_synthesize_from_trace", lambda *_args, **_kwargs: "Yamasaki, Itoh")

    state = {
        "question": "Who are the pitchers before and after Tamai?",
        "attempt_count": 3,
        "messages": [AIMessage(content="Need one more step")],
        "last_error": "",
        "tool_trace": [{"tool": "web_search", "preview": "Title: roster"}],
    }

    result = agent._evidence_check_node(cast(GaiaAgentState, state))
    assert result.get("candidate_answer") == "Yamasaki, Itoh"
    assert result.get("stop_reason") == "synthesized_after_max_iterations"


def test_evidence_check_detects_repeated_identical_signature():
    agent = _make_agent_for_evidence_tests(max_iterations=12)
    repeated_signature = 'execute_python::{"code":"x=1"}'
    state = {
        "question": "dummy question",
        "attempt_count": 4,
        "messages": [AIMessage(content="Trying")],
        "last_error": "",
        "tool_trace": [
            {"tool": "execute_python", "preview": "tool_call", "signature": repeated_signature},
            {"tool": "execute_python", "preview": "tool_call", "signature": repeated_signature},
            {"tool": "execute_python", "preview": "tool_call", "signature": repeated_signature},
        ],
    }

    result = agent._evidence_check_node(cast(GaiaAgentState, state))
    assert result.get("stop_reason") == "stagnation_detected_same_call"
    hint_messages = result.get("messages", [])
    assert hint_messages
    hint_text = str(getattr(hint_messages[0], "content", ""))
    assert "Repeated call signature" in hint_text


def test_evidence_check_stops_early_when_required_attachment_missing():
    agent = _make_agent_for_evidence_tests(max_iterations=12)
    state = {
        "question": "Review the attached image and provide the move.",
        "attempt_count": 1,
        "messages": [AIMessage(content="I still need the file")],
        "last_error": "failed downloading attachment",
        "task_file_status": "no_file",
        "file_name": "position.png",
        "tool_trace": [],
    }

    result = agent._evidence_check_node(cast(GaiaAgentState, state))
    assert result.get("candidate_answer") == "I don't know"
    assert result.get("stop_reason") == "task_file_unavailable"


def test_evidence_check_pushes_image_tool_first_when_image_available():
    agent = _make_agent_for_evidence_tests(max_iterations=12)
    state = {
        "question": "Review this board and find best black move.",
        "attempt_count": 1,
        "messages": [AIMessage(content="Let me search web first")],
        "last_error": "",
        "task_file_status": "downloaded",
        "file_name": "board.png",
        "downloaded_files": ["/tmp/board.png"],
        "tool_trace": [],
    }

    result = agent._evidence_check_node(cast(GaiaAgentState, state))
    assert result.get("stop_reason") == "image_tool_not_used_yet"
    hint_messages = result.get("messages", [])
    assert hint_messages
    hint_text = str(getattr(hint_messages[0], "content", ""))
    assert "analyze_image_with_vlm" in hint_text


def test_task_file_fetch_short_circuits_when_required_attachment_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = _make_agent_for_evidence_tests(max_iterations=12)
    capture: dict[str, str] = {}

    def _stub_download(
        task_id: str,
        api_url: str | None = None,
        destination_dir: str | None = None,
        expected_filename: str = "",
    ):
        capture["task_id"] = task_id
        capture["expected_filename"] = expected_filename
        return {"status": "no_file", "message": "NO_FILE"}

    monkeypatch.setattr("agent.graph.download_task_file_raw", _stub_download)

    state = {
        "task_id": "task-1",
        "question": "Review the attached image and provide the move.",
        "file_name": "task-1.png",
        "working_dir": "/tmp",
        "downloaded_files": [],
    }

    result = agent._task_file_fetch_node(cast(GaiaAgentState, state))
    assert capture.get("task_id") == "task-1"
    assert capture.get("expected_filename") == "task-1.png"
    assert result.get("candidate_answer") == "I don't know"
    assert result.get("stop_reason") == "task_file_unavailable"


def test_download_task_file_raw_uses_dataset_fallback_on_primary_404(
    monkeypatch: pytest.MonkeyPatch,
):
    class _PrimaryNotFoundResponse:
        status_code = 404
        headers = {}
        content = b""

        @staticmethod
        def raise_for_status() -> None:
            raise RuntimeError("should not be called for 404 path")

    monkeypatch.setattr(
        "agent.tools.core.requests.get",
        lambda *args, **kwargs: _PrimaryNotFoundResponse(),
    )

    captured: dict[str, str] = {}

    def _fallback(
        task_id: str,
        destination_dir: str | None = None,
        expected_filename: str = "",
    ):
        captured["task_id"] = task_id
        captured["destination_dir"] = str(destination_dir or "")
        captured["expected_filename"] = expected_filename
        return {
            "status": "downloaded",
            "path": "/tmp/fallback.png",
            "filename": "fallback.png",
            "message": "OK",
            "source": "gaia_dataset",
        }

    monkeypatch.setattr("agent.tools.core._download_task_file_from_gaia_dataset", _fallback)

    result = download_task_file_raw(
        task_id="task-404",
        api_url="https://example.invalid",
        destination_dir="/tmp",
        expected_filename="task-404.png",
    )

    assert captured.get("task_id") == "task-404"
    assert captured.get("destination_dir") == "/tmp"
    assert captured.get("expected_filename") == "task-404.png"
    assert result.get("status") == "downloaded"
    assert result.get("source") == "gaia_dataset"


def test_heuristic_extraction_for_pitcher_before_after():
    question = (
        "Who are the pitchers with the number before and after Taisho Tamai's number? "
        "Use last names only."
    )
    tool_trace = [
        {
            "tool": "web_search",
            "preview": (
                "Hokkaido Nippon-Ham Fighters 18 Pitcher Yamasaki, Sachiya "
                "19 Pitcher Tamai, Taisho 20 Pitcher Itoh, Hiromi"
            ),
        }
    ]

    candidate = GaiaLangGraphAgent._heuristic_answer_from_trace(question, tool_trace)
    assert candidate == "Yamasaki, Itoh"


def test_heuristic_extraction_for_pitcher_before_after_mixed_roster_snippets():
    question = (
        "Who are the pitchers with the number before and after Taisho Tamai's number? "
        "Use last names only."
    )
    tool_trace = [
        {
            "tool": "web_search",
            "preview": (
                "Players ; Pitchers. 12 Kota Yazawa; 17 Hiromi Itoh; "
                "18 Sachiya Yamasaki; 20 Kenta Uehara;"
            ),
        },
        {
            "tool": "web_search",
            "preview": "19 Hokkaido Nippon-Ham Fighters Tamai, Taisho",
        },
    ]

    candidate = GaiaLangGraphAgent._heuristic_answer_from_trace(question, tool_trace)
    assert candidate == "Yamasaki, Uehara"


def test_heuristic_extraction_for_pitcher_before_after_markdown_table():
    question = (
        "Who are the pitchers with the number before and after Taisho Tamai's number? "
        "Use last names only."
    )
    tool_trace = [
        {
            "tool": "web_search",
            "preview": (
                "### Summary Table (2023 Season) | Number | Player | Position | "
                "| :--- | :--- | :--- | | **18** | **Kosei Yoshida** | Pitcher | "
                "| **19** | **Taisho Tamai** | Pitcher | | **20** | **Kenta Uehara** | Pitcher |"
            ),
        }
    ]

    candidate = GaiaLangGraphAgent._heuristic_answer_from_trace(question, tool_trace)
    assert candidate == "Yoshida, Uehara"


def test_heuristic_extraction_for_nasa_award_number():
    question = "Under what NASA award number was the work supported by?"
    tool_trace = [
        {
            "tool": "fetch_webpage_text",
            "preview": "Acknowledgments: R. G. Arendt was supported under NASA award number 80NSSC21K0578.",
        }
    ]

    candidate = GaiaLangGraphAgent._heuristic_answer_from_trace(question, tool_trace)
    assert candidate == "80NSSC21K0578"


def test_heuristic_extraction_for_1928_olympics_least_athletes_ioc_code():
    question = (
        "At the 1928 Summer Olympics, 3 IOC country codes had the least number of athletes. "
        "What is the IOC country code that comes first alphabetically?"
    )
    tool_trace = [
        {
            "tool": "fetch_webpage_text",
            "preview": "ARG 80 - 81 - 81 CUB 2 - 2 - 2 ZIM 2 - 2 - 2",
        }
    ]

    candidate = GaiaLangGraphAgent._heuristic_answer_from_trace(question, tool_trace)
    assert candidate == "CUB"


def test_benchmark_leak_url_blocks_hf_gaia_dataset_discussion():
    url = "https://huggingface.co/datasets/gaia-benchmark/GAIA/discussions/26"
    assert _is_potential_benchmark_leak_url(url) is True


def test_benchmark_leak_content_blocks_gaia_discussion_like_text():
    content = (
        "gaia-benchmark/GAIA · Question a1e91b78-d3d8-4675-bb8d-62741b4b68a6 seems like the wrong answer. "
        "Discussion mentions Final answer and task_id for benchmark questions."
    )
    assert _is_potential_benchmark_leak_content(content) is True


def test_benchmark_leak_url_blocks_hf_sft_dataset():
    url = "https://huggingface.co/datasets/cat-searcher/owl-sft-dataset"
    assert _is_potential_benchmark_leak_url(url) is True


def test_benchmark_leak_url_blocks_gaia_space_commit_with_task_uuid():
    url = (
        "https://huggingface.co/spaces/bstraehle/gaia/commit/"
        "7dae5d7a75a4d33623683fd1a0995f6dcc23679c"
        "?task_id=7bd855d8-463d-4ed5-93ca-5fe35145f733"
    )
    assert _is_potential_benchmark_leak_url(url) is True


def test_benchmark_leak_url_blocks_final_assignment_template_space_commit():
    url = "https://huggingface.co/spaces/agents-course/Final_Assignment_Template/commit/abc123"
    assert _is_potential_benchmark_leak_url(url) is True


def test_benchmark_leak_url_blocks_scoring_questions_endpoint():
    url = "https://agents-course-unit4-scoring.hf.space/questions"
    assert _is_potential_benchmark_leak_url(url) is True


def test_resolve_search_providers_normalizes_aliases_and_dedupes():
    providers = _resolve_search_providers("Google, tavily, duckduckgo, ddg, unknown")
    assert providers == ["google", "tavily", "ddg"]


def test_url_dedupe_key_preserves_meaningful_query_parameters():
    key_a = _url_dedupe_key("https://www.youtube.com/watch?v=abc123&utm_source=test")
    key_b = _url_dedupe_key("https://youtube.com/watch?v=def456&utm_source=other")
    assert key_a != key_b

    key_c = _url_dedupe_key("https://example.com/path?a=1&utm_source=x")
    key_d = _url_dedupe_key("https://example.com/path?a=1&utm_source=y")
    assert key_c == key_d


def test_merge_and_rank_search_hits_prefers_multi_provider_overlap():
    hits_by_provider = {
        "google": [
            {
                "title": "Official reference",
                "url": "https://example.com/resource/",
                "snippet": "Official result from provider A.",
                "provider": "google",
            }
        ],
        "tavily": [
            {
                "title": "Official reference (mirror)",
                "url": "https://example.com/resource",
                "snippet": "Official result from provider B with richer snippet content.",
                "provider": "tavily",
            }
        ],
        "ddg": [
            {
                "title": "Forum discussion",
                "url": "https://reddit.com/r/example/comments/abc123",
                "snippet": "Community thread with uncertain answer quality.",
                "provider": "ddg",
            }
        ],
    }

    ranked = _merge_and_rank_search_hits(hits_by_provider=hits_by_provider, max_items=5)
    assert ranked
    assert str(ranked[0]["url"]).startswith("https://example.com/resource")
    assert set(ranked[0]["providers"]) == {"google", "tavily"}


def test_web_search_falls_back_to_ddg_when_selected_providers_return_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "agent.tools.core._run_parallel_provider_search",
        lambda query, max_items, providers, timeout_seconds: ({}, {}),
    )
    monkeypatch.setattr(
        "agent.tools.core._search_ddg_hits",
        lambda query, max_items, timeout_seconds=8: [
            {
                "title": "DDG fallback hit",
                "url": "https://example.org/fallback",
                "snippet": "Fallback snippet",
                "provider": "ddg",
            }
        ],
    )

    result = web_search.invoke(
        {
            "query": "fallback query",
            "max_results": 3,
            "providers": "google,tavily",
        }
    )

    assert "https://example.org/fallback" in result
    assert "Provider(s): ddg" in result


def test_web_search_surfaces_provider_warnings_with_partial_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "agent.tools.core._run_parallel_provider_search",
        lambda query, max_items, providers, timeout_seconds: (
            {
                "google": [
                    {
                        "title": "Good result",
                        "url": "https://example.com/good",
                        "snippet": "Useful snippet",
                        "provider": "google",
                    }
                ]
            },
            {"tavily": "RuntimeError: missing TAVILY_API_KEY"},
        ),
    )

    result = web_search.invoke(
        {
            "query": "provider warning behavior",
            "max_results": 3,
            "providers": "google,tavily",
        }
    )

    assert "https://example.com/good" in result
    assert "Provider warnings:" in result
    assert "tavily" in result


def test_google_search_provider_prefers_genai_hits(monkeypatch: pytest.MonkeyPatch):
    cse_called = {"value": False}

    monkeypatch.setattr(
        "agent.tools.core._search_google_genai_hits",
        lambda query, max_items, timeout_seconds=8: [
            {
                "title": "GenAI grounded source",
                "url": "https://example.com/genai",
                "snippet": "Grounded snippet",
                "provider": "google",
            }
        ],
    )

    def _cse_stub(query: str, max_items: int, timeout_seconds: int = 8):
        cse_called["value"] = True
        return [
            {
                "title": "CSE fallback source",
                "url": "https://example.com/cse",
                "snippet": "CSE snippet",
                "provider": "google",
            }
        ]

    monkeypatch.setattr("agent.tools.core._search_google_cse_hits", _cse_stub)

    hits = _search_google_custom_search_hits(query="test", max_items=3, timeout_seconds=5)
    assert hits
    assert hits[0]["url"] == "https://example.com/genai"
    assert cse_called["value"] is False


def test_google_search_provider_falls_back_to_cse_when_genai_errors(monkeypatch: pytest.MonkeyPatch):
    def _genai_raises(*_args, **_kwargs):
        raise RuntimeError("genai unavailable")

    monkeypatch.setattr("agent.tools.core._search_google_genai_hits", _genai_raises)
    monkeypatch.setattr(
        "agent.tools.core._search_google_cse_hits",
        lambda query, max_items, timeout_seconds=8: [
            {
                "title": "CSE fallback source",
                "url": "https://example.com/cse",
                "snippet": "CSE snippet",
                "provider": "google",
            }
        ],
    )

    hits = _search_google_custom_search_hits(query="fallback", max_items=3, timeout_seconds=5)
    assert hits
    assert hits[0]["url"] == "https://example.com/cse"


def test_evidence_check_uses_synthesis_when_search_budget_exhausted(monkeypatch: pytest.MonkeyPatch):
    agent = _make_agent_for_evidence_tests(max_iterations=10)
    monkeypatch.setattr(agent, "_synthesize_from_trace", lambda *_args, **_kwargs: "CUB")

    state = {
        "question": "At the 1928 Summer Olympics, what IOC code comes first alphabetically among the least athletes?",
        "attempt_count": 8,
        "messages": [AIMessage(content="Need one more lookup")],
        "last_error": "",
        "tool_trace": [
            {"tool": "web_search", "preview": "tool_call"},
            {"tool": "web_search", "preview": "tool_call"},
            {"tool": "web_search", "preview": "tool_call"},
        ],
    }

    result = agent._evidence_check_node(cast(GaiaAgentState, state))
    assert result.get("candidate_answer") == "CUB"
    assert result.get("stop_reason") == "synthesized_after_search_budget"
