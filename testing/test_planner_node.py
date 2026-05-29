from unittest.mock import MagicMock
import pytest
from agents.planner_agent import run_planner_node, PlannerOutput


def _make_llm(plan="Test plan.", sub_questions=None, search_queries=None):
    """Helper: LLM mock that returns a PlannerOutput directly."""
    output = PlannerOutput(
        plan=plan,
        sub_questions=sub_questions if sub_questions is not None else ["Q1", "Q2", "Q3", "Q4"],
        search_queries=search_queries if search_queries is not None else ["query 1", "query 2", "query 3", "query 4"],
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = output

    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


# ── Normal case ───────────────────────────────────────────────────────────────

def test_planner_returns_all_fields():
    llm = _make_llm()
    result = run_planner_node({"topic": "quantum computing"}, llm)

    assert result["plan"] == "Test plan."
    assert len(result["sub_questions"]) == 4
    assert len(result["search_queries"]) == 4
    assert result["search_topic"] == "query 1"           # first query becomes primary
    assert result["rewritten_queries"] == result["search_queries"]  # backward-compat


# ── Empty topic ───────────────────────────────────────────────────────────────

def test_planner_empty_topic_skips_llm():
    llm = MagicMock()
    result = run_planner_node({"topic": ""}, llm)

    llm.with_structured_output.assert_not_called()      # LLM never invoked
    assert result["plan"] == ""
    assert result["sub_questions"] == []
    assert result["search_queries"] == []


# ── Fallback when LLM returns empty lists ─────────────────────────────────────

def test_planner_fallback_when_empty_queries():
    llm = _make_llm(sub_questions=[], search_queries=[])
    result = run_planner_node({"topic": "AI safety"}, llm)

    # Falls back to raw topic
    assert result["search_queries"] == ["AI safety"]
    assert result["sub_questions"] == ["AI safety"]


# ── LLM exception ─────────────────────────────────────────────────────────────

def test_planner_handles_llm_exception():
    llm = MagicMock()
    llm.with_structured_output.side_effect = RuntimeError("API down")

    result = run_planner_node({"topic": "climate change"}, llm)

    assert "error" in result
    assert result["search_queries"] == ["climate change"]  # graceful fallback
    assert result["plan"] == ""