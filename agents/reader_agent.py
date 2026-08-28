"""
agents/reader_agent.py
────────────────────────────────────────────────────────────
Reads ALL verified URLs in true parallel using ThreadPoolExecutor.

Key design decisions:
  - ALL URLs scraped simultaneously (not one-by-one, not just top-3)
  - Time taken = slowest single URL, not sum of all URLs
  - Cap at MAX_PARALLEL_URLS to keep writer context manageable
  - Per-URL content capped at 2500 chars to avoid token overflow
"""

import logging
from tools import extract_urls_from_search_output
from tools.scraper import scrape_many_parallel

log = logging.getLogger(__name__)

MAX_PARALLEL_URLS = 8       # scrape up to 8 URLs simultaneously
MAX_CHARS_PER_URL = 2500    # cap per-page content


def build_reader_agent():
    """
    Reader does deterministic parsing + parallel scraper tool call.
    Kept as a builder for interface consistency.
    """
    return {"name": "parallel_reader_v2"}


def run_reader_agent(state: dict, _reader) -> dict:
    """
    Extract ALL URLs from search output and scrape them ALL in parallel.

    All HTTP requests fire simultaneously via ThreadPoolExecutor.
    Total time = slowest single page (~4-6s), not sum of all pages.
    """
    if state.get("error"):
        return {**state, "scraped_content": ""}

    # Collect all verified URLs from the searcher
    verified_urls = state.get("verified_urls", [])
    if verified_urls:
        urls = verified_urls[:MAX_PARALLEL_URLS]
    else:
        search_text = state.get("search_results", "")
        if isinstance(search_text, list):
            search_text = "\n\n".join(search_text)
        urls = extract_urls_from_search_output(search_text, top_k=MAX_PARALLEL_URLS)

    if not urls:
        return {
            **state,
            "urls": [],
            "scraped_content": "",
            "error": state.get("error") or "No URLs found in search output.",
        }

    log.info(
        "Reader: scraping %d URLs in parallel (max_workers=%d)...",
        len(urls), len(urls),
    )

    try:
        # All URLs scraped simultaneously — time = max(individual times)
        results_by_url: dict = scrape_many_parallel(
            urls=urls,
            max_workers=len(urls),   # all in parallel, no queuing
        )

        blocks = []
        success_count = 0
        for url in urls:
            content = results_by_url.get(url, "")
            if content and not content.startswith("ERROR"):
                # Cap per-URL content to avoid writer token overflow
                blocks.append(f"Source: {url}\n{content[:MAX_CHARS_PER_URL]}")
                success_count += 1
            else:
                log.warning("Reader: failed to scrape %s — %s", url, content[:100] if content else "empty")

        combined = "\n\n".join(blocks)
        log.info(
            "Reader: %d/%d URLs scraped successfully | %d chars total",
            success_count, len(urls), len(combined),
        )
        return {**state, "urls": urls, "scraped_content": combined}

    except Exception as exc:
        log.exception("Reader failed")
        return {
            **state,
            "urls": urls,
            "scraped_content": "",
            "error": f"Reader failed: {exc}",
        }