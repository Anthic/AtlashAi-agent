"""
pipeline/__init__.py
────────────────────────────────────────────────────────────
Singleton pipeline cache + public run_research() entrypoint.
Auto-saves every completed result to research history.
"""

import time
from typing import Generator
from .runner import build_pipeline

_pipeline_cache : dict = {}


def _get_pipeline(mode: str = "deep"):
    if mode not in _pipeline_cache:
        _pipeline_cache[mode] = build_pipeline(mode=mode)
    return _pipeline_cache[mode]




def _make_initial_state(topic: str , mode : str = "deep") -> dict:
    return {
        "topic": topic,
        "mode" : mode,
        # Planner output
        "plan": "",
        "sub_questions": [],
        "search_queries": [],
        # Query rewrite
        "rewritten_queries": [],
        "search_topic": "",
        # Search
        "search_results": [],
        "verified_urls": [],
        "urls": [],
        # Scrape + summarise
        "scraped_content": "",
        "summarized_content": "",
        # RAG
        "rag_context": "",
        # Writer / critic
        "report": "",
        "critique": "",
        "critique_score": 0,
        "retry_count": 0,
        "max_retries": 1,
        # Fact-check
        "fact_check_result": "",
        "fact_check_score": 0.0,
        # Meta
        "error": "",
    }


# ── Streaming generator (new) ────────────────────────────────────────────
def stream_research(topic: str, job_id: str | None = None, mode: str = "deep") -> Generator[dict, None, None]:
    """
    Yields an event after each node finishes in LangGraph.

    Event shape:
      {"node_name": {"changed_key": value, ...}}   
      {"__done__": {...full result...}}            
    """
    pipeline = _get_pipeline(mode=mode)
    initial_state = _make_initial_state(topic ,mode=mode)

    start = time.time()
    final_state = {}

    for event in pipeline.stream(initial_state, stream_mode="updates"):
        # event = {"node_name": {changed_keys...}}
        # Accumulate into final_state to build summary at the end
        node_data = next(iter(event.values()))
        final_state.update(node_data)
        yield event  # ← caller / SSE endpoint receives this

    elapsed = round(time.time() - start, 2)

    # After all nodes finish, yield a summary event
    yield {
        "__done__": {
            "job_id": job_id,
            "topic": final_state.get("topic", topic),
            "report": final_state.get("report", ""),
            "critique": final_state.get("critique", ""),
            "critique_score": final_state.get("critique_score", 0),
            "fact_check_score": final_state.get("fact_check_score") or 0.0,
            "fact_check_result": final_state.get("fact_check_result", ""),
            "rewritten_queries": final_state.get("rewritten_queries", []),
            "verified_urls": final_state.get("verified_urls", []),
            "error": final_state.get("error", ""),
            "time_sec": elapsed,
        }
    }


# ── Blocking version — old interface unchanged ─────────────────────────────
def run_research(topic: str, job_id: str | None = None, mode: str = "deep") -> dict:
    """
    Consumes stream_research() and returns a single dict.
    No changes needed in old code that calls run_research().
    """
    result = {}

    for event in stream_research(topic, job_id, mode=mode):
        if "__done__" in event:
            result = event["__done__"]

    # Persist to history (best-effort — never crash main flow)
    try:
        from memory import save_research
        save_research(result)
    except Exception:
        pass

    return result


__all__ = ["build_pipeline", "run_research", "stream_research"]