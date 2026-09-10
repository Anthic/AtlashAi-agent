import os
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tools.cache_tool import cached_search

load_dotenv()
log = logging.getLogger(__name__)

YDC_API_KEY = os.getenv("YDC_API_KEY", "")
YDC_SEARCH_URL = "https://ydc-index.io/v1/search"


def you_web_search(query: str, num_web_results: int = 5) -> List[Dict[str, Any]]:
    """
    Direct caller for You.com Search API with Upstash Redis caching.
    Returns a list of dicts: [{"title": ..., "url": ..., "snippets": [...], "description": ...}]
    """
    if not YDC_API_KEY:
        log.warning("YDC_API_KEY is not set. Skipping You.com search.")
        return []

    clean_query = query.strip()
    if not clean_query:
        return []

    def _fetch_from_api(_: str) -> dict:
        try:
            headers = {"X-API-Key": YDC_API_KEY}
            params = {
                "query": clean_query,
                "num_web_results": num_web_results
            }
            resp = requests.get(YDC_SEARCH_URL, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                log.error(f"You.com API error {resp.status_code}: {resp.text}")
                return {"results": {"web": []}}
        except Exception as e:
            log.exception(f"Exception during You.com API call: {str(e)}")
            return {"results": {"web": []}}

    cached_data = cached_search(f"you:{clean_query}", _fetch_from_api, ttl=3600)

    # You.com returns data inside results -> web, or legacy hits
    hits = []
    if isinstance(cached_data, dict):
        hits = cached_data.get("results", {}).get("web", []) or cached_data.get("hits", [])

    results = []
    for hit in hits:
        snippets = hit.get("snippets", [])
        desc = hit.get("description", "")
        if not snippets and desc:
            snippets = [desc]
        results.append({
            "title": hit.get("title", ""),
            "url": hit.get("url", ""),
            "snippets": snippets,
            "description": desc
        })

    return results


def search_temporal_facts(claim: str, topic: str = "") -> List[Dict[str, Any]]:
    """
    Searches for current state, benchmarks, or updates regarding a specific claim (2024-2026).
    """
    query = f"{claim} benchmark latest state of the art controversy 2024 2025 2026"
    if topic:
        query = f"{topic}: {query}"
    return you_web_search(query, num_web_results=3)




def search_academic_critiques(topic: str, methodology: str = "") -> List[Dict[str, Any]]:
    """
    Searches for real-world critiques, limitations, or vulnerabilities of a method.
    """
    query = f"{topic} {methodology} limitations critique failure modes Reddit arXiv"
    return you_web_search(query, num_web_results=4)
    