
from typing import List, TypedDict, Dict


class ResearchState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    topic: str
    mode: str                       # "fast" | "deep" (must be declared or LangGraph drops it)

    # ── Planner output ─────────────────────────────────────────────────────
    plan: str                       
    sub_questions: List[str]        
    search_queries: List[str]       

    # ── Query rewrite (legacy – kept for backward-compat) ──────────────────
    rewritten_queries: List[str]
    search_topic: str               

    # ── Search / Scrape ────────────────────────────────────────────────────
    # Plain overwrite. Every node returns {**state, ...}, so an operator.add
    # reducer appended the list to itself at each node (1 -> 64 items).
    search_results: List[str]
    verified_urls: List[str]
    urls: List[str]

    # ── Summarize ──────────────────────────────────────────────────────────
    scraped_content: str
    summarized_content: str

    # ── RAG ────────────────────────────────────────────────────────────────
    rag_context: str

    # ── Writer ────────────────────────────────────────────────────────────
    report: str
    draft_sections: List[str]           
    # graph
    knowledge_graph: Dict
    knowledge_graph_md: str 
    # ── Critic ────────────────────────────────────────────────────────────
    critique: str
    critique_score: int             
    retry_count: int
    max_retries: int

    # ── Fact-check ────────────────────────────────────────────────────────
    fact_check_result: str
    fact_check_score: float
    fact_check_parsed: bool         # False => score is NOT a real verification result

    # ── Meta ──────────────────────────────────────────────────────────────
    error: str

    time_sec: float