from unittest.mock import MagicMock
import pytest
from agents.knowledge_graph_agent import (
    run_knowledge_graph_node,
    KnowledgeGraph,
    Node,
    Edge,
)


def _make_llm(graph: KnowledgeGraph = None):
    """Helper: LLM mock that returns a KnowledgeGraph directly."""
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = graph

    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


def _sample_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(id="1", label="Quantum Computing", type="topic"),
            Node(id="2", label="Qubit",             type="concept"),
            Node(id="3", label="IBM",               type="organization"),
        ],
        edges=[
            Edge(source="1", target="2", relation="uses"),
            Edge(source="3", target="1", relation="develops"),
        ],
    )


# ── Test 1: Normal case ───────────────────────────────────────────────────────

def test_kg_returns_nodes_and_edges():
    llm    = _make_llm(_sample_graph())
    state  = {"report": "Quantum computing uses qubits. IBM develops quantum computers."}
    result = run_knowledge_graph_node(state, llm)

    kg = result["knowledge_graph"]

    # dict হয়েছে কিনা
    assert isinstance(kg, dict)

    # nodes ও edges key আছে কিনা
    assert "nodes" in kg
    assert "edges" in kg

    # সঠিক সংখ্যা
    assert len(kg["nodes"]) == 3
    assert len(kg["edges"]) == 2

    # একটা node এর structure ঠিক আছে কিনা
    assert kg["nodes"][0] == {"id": "1", "label": "Quantum Computing", "type": "topic"}


# ── Test 2: Empty report — LLM call হওয়া উচিত না ────────────────────────────

def test_kg_empty_report_skips_llm():
    llm    = MagicMock()
    result = run_knowledge_graph_node({"report": ""}, llm)

    llm.with_structured_output.assert_not_called()
    assert result["knowledge_graph"] == {}
    assert result["knowledge_graph_md"] == ""


# ── Test 3: Markdown এ node label গুলো আছে কিনা ──────────────────────────────

def test_kg_markdown_contains_labels():
    llm    = _make_llm(_sample_graph())
    state  = {"report": "Some research report about quantum computing."}
    result = run_knowledge_graph_node(state, llm)

    md = result["knowledge_graph_md"]

    # প্রতিটি node label markdown এ থাকা উচিত
    assert "Quantum Computing" in md
    assert "Qubit"             in md
    assert "IBM"               in md

    # edge relation গুলোও থাকা উচিত
    assert "uses"              in md
    assert "develops"          in md


# ── Test 4: LLM exception — graceful fallback ─────────────────────────────────

def test_kg_handles_llm_exception():
    llm = MagicMock()
    llm.with_structured_output.side_effect = RuntimeError("API timeout")

    state  = {"report": "Some research report."}
    result = run_knowledge_graph_node(state, llm)

    # pipeline বন্ধ হওয়া উচিত না
    assert result["knowledge_graph"]    == {}
    assert result["knowledge_graph_md"] == ""

    # error key থাকা উচিত
    assert "error" in result
    assert "KnowledgeGraphAgent failed" in result["error"]