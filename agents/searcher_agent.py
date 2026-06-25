import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools import tavily_search_tool
from tools.cache_tool import cached_search

log = logging.getLogger(__name__)

MAX_PARALLEL_WORKERS = 8


def run_searcher_node(state: dict, search_agent) -> dict:
    """
    LangGraph node — searches for every query from the planner.
    Runs queries in parallel using ThreadPoolExecutor and direct Tavily calls,
    integrated with Upstash Redis caching.
    """
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

    def _run_one(query: str, idx: int) -> tuple:
        log.info("SearcherAgent: running query %d/%d — %r", idx + 1, len(queries), query)
        
        # Direct Tavily Search function to pass to cached_search
        def _search(q: str) -> dict:
            try:
                # Direct Tavily call bypassing the ReAct LLM agent
                return tavily_search_tool.invoke(q)
            except Exception as e:
                log.error("Tavily search failed for query %r: %s", q, e)
                return {"results": []}

        # Use Upstash Redis cached search
        res = cached_search(query, _search, ttl=3600)
        
        results = res.get("results", []) if isinstance(res, dict) else []
        
        snippets = []
        local_urls = []
        for item in results:
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "")
            if url:
                local_urls.append(url)
            snippets.append(f"Title: {title}\nURL: {url}\nSnippet: {content}")
        
        snippet_text = "\n".join(snippets)
        return idx, snippet_text, local_urls

    worker_count = min(len(queries), MAX_PARALLEL_WORKERS)
    results = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_run_one, q, i): i
            for i, q in enumerate(queries)
        }
        for future in as_completed(futures):
            try:
                idx, snippet, urls = future.result(timeout=60)
                results[idx] = (snippet, urls)
            except Exception as exc:
                i = futures[future]
                log.warning("SearcherAgent: query %d failed — %s", i, exc)
                results[i] = ("", [])

    for i, res in enumerate(results):
        if res is None:
            continue
        snippet, urls = res
        if snippet:
            all_snippets.append(f"[Query {i+1}: {queries[i]}]\n{snippet}")
        for url in urls:
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
        "urls": list(all_urls),
        "topic": state.get("topic", ""),
    }