"""
pipeline/runner.py  (upgraded)
────────────────────────────────────────────────────────────
Full pipeline with new Planner + Searcher nodes:
 
  START
    → planner       ← NEW: builds plan, sub-questions, search queries
    → searcher      ← NEW: runs ALL planner queries (multi-query search)
    → reader        ← parallel scrape verified URLs
    → summarize     ← condense noisy scraped content
    → rag           ← Qdrant vector similarity retrieval
    → writer        ← write grounded report  (uses plan as outline)
    → fact_check    ← verify claims in report
    → critic        ← score & feedback
    → END  (or rewrite loop back to writer)
 
Key improvements over v1:
  • PlannerAgent decomposes topic → sub-questions → search queries
  • SearcherAgent runs one Tavily search per planner query (multi-query)
  • WriterNode receives the plan so the report follows the intended outline
  • Conditional rewrite loop unchanged (score < 8 → retry up to max_retries)
"""

import logging
import re
from typing import List
from agents.knowledge_graph_agent import run_knowledge_graph_node 
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
 
from agents import (
    build_search_agent,
    run_search_agent,
    build_reader_agent,
    run_reader_agent,
    run_query_rewrite_node,
    run_summarizer_node,
    run_fact_check_node,
)
from agents.planner_agent import run_planner_node
from agents.searcher_agent import run_searcher_node
from pipeline.model import get_llm
from pipeline.rag import run_rag_node
from pipeline.chains import (
    build_writer_chain,
    build_critic_chain,
    run_critic,
)
from state import ResearchState
 
load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)
 
 
# ── Score parser ──────────────────────────────────────────────────────────────
 
def parse_score(critique: str) -> int:
    match = re.search(
        r"score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*10",
        critique or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return 0
    return max(0, min(10, round(float(match.group(1)))))
 
 
# ── Router ────────────────────────────────────────────────────────────────────
 
def route_after_critic(state: ResearchState) -> str:
    """
    Conditional edge after critic node.
 
    score >= 8      → "end"    (good enough)
    retries left    → "rewrite" (send back to writer with critique)
    retries exhausted → "end"  (give up gracefully)
    """
    score    = state.get("critique_score", 0)
    retries  = state.get("retry_count", 0)
    max_ret  = state.get("max_retries", 1)
 
    if score >= 8:
        log.info("Router → END  (score=%d)", score)
        return "end"
    if retries < max_ret:
        log.info("Router → REWRITE  (score=%d, retry %d/%d)", score, retries + 1, max_ret)
        return "rewrite"
    log.info("Router → END  (max retries reached, score=%d)", score)
    return "end"
 
 
def bump_retry(state: ResearchState) -> ResearchState:
    """Increment retry counter before sending back to writer."""
    return {**state, "retry_count": state.get("retry_count", 0) + 1}
 
 
# ── Node wrappers ─────────────────────────────────────────────────────────────
 
def run_critic_node(state: dict, critic_chain) -> dict:
    """Run critic and parse numeric score into state."""
    truncated = {**state, "report": state.get("report", "")[:3000]}
    updated = run_critic(truncated, critic_chain)
    updated["critique_score"] = parse_score(updated.get("critique", ""))
    return updated
 
 
def run_writer_node(state: dict, writer_chain) -> dict:
    """
    Upgraded writer that injects the planner's outline into the prompt.
    Falls back gracefully if plan is absent (backward-compat).
    """
    from pipeline.chains import run_writer
 
    # Inject plan into critique_section so the writer uses it as structure hint
    plan = state.get("plan", "").strip()
    existing_critique = state.get("critique", "").strip()
 
    hints = []
    if plan:
        hints.append(f"Research Plan (follow this outline):\n{plan}")
    if existing_critique:
        hints.append(f"Previous critique — improve based on this feedback:\n{existing_critique}")
 
    patched_state = {
        **state,
        # We pass the plan hint via critique so run_writer picks it up
        # without changing the chain signature.
        "critique": "\n\n".join(hints),
    }
    result = run_writer(patched_state, writer_chain)
    # Restore original critique so the critic sees the real previous feedback
    result["critique"] = existing_critique
    return result
 
 
# ── Pipeline builder ──────────────────────────────────────────────────────────
 
def build_pipeline():
    fast_llm  = get_llm("fast")
    smart_llm = get_llm("smart")
 
    search_agent  = build_search_agent(fast_llm)
    reader_agent  = build_reader_agent()
    writer_chain  = build_writer_chain(smart_llm)
    critic_chain  = build_critic_chain(fast_llm)
 
    graph = StateGraph(ResearchState)
 
    # ── Nodes ──────────────────────────────────────────────────────────────
    # NEW: Planner — decomposes topic → plan + sub-questions + search queries
    graph.add_node("planner",
        lambda s: run_planner_node(s, fast_llm))
 
    # NEW: Searcher — runs multi-query Tavily search using planner queries
    graph.add_node("searcher",
        lambda s: run_searcher_node(s, search_agent))
 
    # Existing nodes (unchanged internals)
    graph.add_node("reader",
        lambda s: run_reader_agent(s, reader_agent))
    graph.add_node("summarize",
        lambda s: run_summarizer_node(s, fast_llm))
    graph.add_node("rag",
        lambda s: run_rag_node(s))
 
    # Upgraded writer (injects plan)
    graph.add_node("writer",
        lambda s: run_writer_node(s, writer_chain))
 
    graph.add_node("fact_check",
        lambda s: run_fact_check_node(s, fast_llm))
 
    # Upgraded critic (scores + routes)
    graph.add_node("critic",
        lambda s: run_critic_node(s, critic_chain))
    
    graph.add_node("knowledge_graph",
        lambda s: run_knowledge_graph_node(s, smart_llm))

    graph.add_node("prepare_rewrite", bump_retry)
 
    # ── Edges ──────────────────────────────────────────────────────────────
    graph.add_edge(START,            "planner")    
    graph.add_edge("planner",        "searcher")   
    graph.add_edge("searcher",       "reader")
    graph.add_edge("reader",         "summarize")
    graph.add_edge("summarize",      "rag")
    graph.add_edge("rag",            "writer")
    graph.add_edge("writer",         "fact_check")
    graph.add_edge("fact_check",     "critic")

    # ── Conditional rewrite loop ───────────────────────────────────────────
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "rewrite": "prepare_rewrite",
            "end":     "knowledge_graph",
        },
    )
    graph.add_edge("prepare_rewrite", "writer")
    graph.add_edge("knowledge_graph", END) 
 
    return graph.compile()