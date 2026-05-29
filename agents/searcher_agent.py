import logging
from typing import Dict, List

from agents.search_agent import run_search_agent  # module-level so tests can patch it

log = logging.getLogger(__name__)


def run_searcher_node(state: dict, search_agent) -> dict:
    """
    LangGraph node — searches for every query from the planner.

    Falls back to the single-query path (search_topic / topic) if
    search_queries is empty so the pipeline stays backward-compatible.
    """
    # run_search_agent imported at module level above

    queries: List[str] = state.get("search_queries", [])

    # ── Fallback to legacy single query ──────────────────────────────────
    if not queries:
        effective = state.get("search_topic") or state.get("topic", "")
        queries = [effective] if effective else []

    if not queries:
        log.warning("SearcherAgent: no queries to run.")
        return {
            **state,
            "search_results": [],
            "verified_urls": [],
            "urls": [],
            "error": "SearcherAgent: no queries provided.",
        }

    all_snippets: List[str] = []
    all_urls: List[str] = []
    seen_urls: set = set()

    for i, query in enumerate(queries):
        log.info("SearcherAgent: running query %d/%d — %r", i + 1, len(queries), query)


        patched_state = {
            **state,
            "topic": query,           
            "search_results": [],
            "verified_urls": [],
            "urls": [],
        }
        result = run_search_agent(patched_state, search_agent)

        snippet = result.get("search_results", "").strip()
        if snippet:
            all_snippets.append(f"[Query {i+1}: {query}]\n{snippet}")

        for url in result.get("verified_urls", []) + result.get("urls", []):
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_urls.append(url)

    combined_results = "\n\n".join(all_snippets)
    log.info(
        "SearcherAgent: %d queries done | %d unique URLs | %d chars",
        len(queries), len(all_urls), len(combined_results),
    )

    return {
        **state,
        "search_results": [combined_results],
        "verified_urls": all_urls,
        "urls": list(all_urls),  # separate copy to avoid shared reference
        "topic": state.get("topic", ""),
    }