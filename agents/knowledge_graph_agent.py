import logging
from typing import List, Literal
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Part A: Pydantic Models ───────────────────────────────────────────────────

class Node(BaseModel):
    id: str
    label: str
    type: Literal["topic", "concept", "organization", "person", "technology"]

class Edge(BaseModel):
    source: str = Field(description="id of the source node")
    target: str = Field(description="id of the target node")
    relation: str = Field(description="verb phrase, e.g. 'uses', 'develops', 'is part of'")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="Max 15 nodes")
    edges: List[Edge] = Field(description="Max 20 edges")

# ── Part B: Prompt ────────────────────────────────────────────────────────────

_KG_PROMPT = """\
You are a knowledge graph expert.

Given a research report, extract:
1. ENTITIES — all important concepts, people, organizations, and technologies \
(max 15 nodes). Assign each a short unique id (e.g. "1", "2", ...).
2. RELATIONSHIPS — directional connections between entities \
(max 20 edges). Use short verb phrases for relation (e.g. "uses", "develops", "is part of").

Rules:
- Every edge's source and target must refer to a valid node id.
- Prefer specific relation labels over generic ones like "related to".
- Do not repeat the same node or edge.

Report:
{report}
"""

# ── Part C: Helper + Node function ───────────────────────────────────────────

def _graph_to_markdown(graph: KnowledgeGraph) -> str:
    """Convert KnowledgeGraph to a readable Markdown string."""
    lines = ["## Knowledge Graph", "", "### Nodes", ""]
    for node in graph.nodes:
        lines.append(f"- **{node.label}** (id: `{node.id}`, type: {node.type})")

    id_to_label = {n.id: n.label for n in graph.nodes}  
    lines += ["", "### Edges", ""]
    for edge in graph.edges:
        src = id_to_label.get(edge.source, edge.source)
        tgt = id_to_label.get(edge.target, edge.target)
        lines.append(f"- **{src}** → _{edge.relation}_ → **{tgt}**")

    return "\n".join(lines)


def run_knowledge_graph_node(state: dict, llm) -> dict:
    """
    LangGraph node.

    Reads  : state['report']
    Writes : state['knowledge_graph']    (Dict — JSON serializable)
             state['knowledge_graph_md'] (Markdown string)
    """
    report = state.get("report", "").strip()

    if not report:
        log.warning("KnowledgeGraphAgent: empty report, skipping.")
        return {**state, "knowledge_graph": {}, "knowledge_graph_md": ""}

    try:
        structured_llm = llm.with_structured_output(KnowledgeGraph)
        prompt = _KG_PROMPT.format(report=report)
        graph: KnowledgeGraph = structured_llm.invoke(prompt)

        graph_dict = graph.model_dump()
        graph_md = _graph_to_markdown(graph)

        log.info(
            "KnowledgeGraphAgent: %d nodes | %d edges",
            len(graph.nodes),
            len(graph.edges),
        )

        return {
            **state,
            "knowledge_graph": graph_dict,
            "knowledge_graph_md": graph_md,
        }

    except Exception as exc:
        log.exception("KnowledgeGraphAgent failed")
        return {
            **state,
            "knowledge_graph": {},
            "knowledge_graph_md": "",
            "error": f"KnowledgeGraphAgent failed: {exc}",
        }