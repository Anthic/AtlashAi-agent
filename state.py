
import operator 
from typing import Annotated
from typing import List, TypedDict, Dict


class ResearchState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    topic: str

    # ── Planner output ─────────────────────────────────────────────────────
    plan: str                       
    sub_questions: List[str]        
    search_queries: List[str]       

    # ── Query rewrite (legacy – kept for backward-compat) ──────────────────
    rewritten_queries: List[str]
    search_topic: str               

    # ── Search / Scrape ────────────────────────────────────────────────────
    search_results: Annotated[List[str], operator.add]
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

    # ── Meta ──────────────────────────────────────────────────────────────
    error: str

    time_sec: float