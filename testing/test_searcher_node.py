from unittest.mock import MagicMock, patch
import pytest
from agents.searcher_agent import run_searcher_node


def _make_search_result(snippet="some result", urls=None):
    return {
        "search_results": snippet,
        "verified_urls": urls or [],
        "urls": urls or [],
    }


# ── Multiple queries loop করছে কিনা ──────────────────────────────────────────

def test_searcher_runs_all_queries():
    state = {
        "topic": "original topic",
        "search_queries": ["query A", "query B", "query C"],
    }

    with patch("agents.searcher_agent.run_search_agent", return_value=_make_search_result()) as mock_search:
        result = run_searcher_node(state, MagicMock())

    assert mock_search.call_count == 3


# ── patched_state topic override হচ্ছে, original topic নষ্ট হচ্ছে না ──────────

def test_searcher_restores_original_topic():
    state = {
        "topic": "original topic",
        "search_queries": ["query A", "query B"],
    }

    with patch("agents.searcher_agent.run_search_agent", return_value=_make_search_result()):
        result = run_searcher_node(state, MagicMock())

    assert result["topic"] == "original topic"   # original restored


# ── Duplicate URL বাদ যাচ্ছে কিনা ────────────────────────────────────────────

def test_searcher_deduplicates_urls():
    state = {"topic": "ML", "search_queries": ["q1", "q2"]}

    # দুটো query তেই একই URL আসছে
    with patch(
        "agents.searcher_agent.run_search_agent",
        return_value=_make_search_result(urls=["https://example.com"]),
    ):
        result = run_searcher_node(state, MagicMock())

    assert result["verified_urls"].count("https://example.com") == 1  # শুধু একবার


# ── search_queries খালি থাকলে search_topic fallback ─────────────────────────

def test_searcher_fallback_to_search_topic():
    state = {
        "topic": "original",
        "search_queries": [],
        "search_topic": "fallback query",
    }

    with patch("agents.searcher_agent.run_search_agent", return_value=_make_search_result()) as mock_search:
        result = run_searcher_node(state, MagicMock())

    # fallback query দিয়ে একবার search হওয়া উচিত
    assert mock_search.call_count == 1
    call_state = mock_search.call_args[0][0]
    assert call_state["topic"] == "fallback query"


# ── কোনো query না থাকলে error return করছে কিনা ──────────────────────────────

def test_searcher_no_queries_returns_error():
    state = {"topic": "", "search_queries": []}

    result = run_searcher_node(state, MagicMock())

    assert "error" in result
    assert result["search_results"] == ""