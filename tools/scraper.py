from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests
import threading
import ipaddress
import socket
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_core.tools import tool

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Lazy-init a thread-local requests Session for thread safety."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update(DEFAULT_HEADERS)
    return _thread_local.session

MAX_REDIRECTS = 4
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024      # HTML pages: 2 MB
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata"}


def _ip_is_public(ip_str: str) -> bool:
    """True only for globally routable addresses (blocks private, loopback, link-local,
    CGNAT 100.64/10, multicast, reserved, unspecified and IPv4-mapped IPv6 forms)."""
    try:
        ip = ipaddress.ip_address(ip_str.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global) and not (ip.is_multicast or ip.is_unspecified or ip.is_loopback or ip.is_link_local)


def is_safe_url(url: str) -> bool:
    """SSRF guard: http(s) only, resolve the host and require EVERY address to be public."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        host = p.hostname.lower().rstrip(".")
        if host in _BLOCKED_HOSTS or host.endswith((".local", ".internal", ".localhost")):
            return False
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
        if not infos:
            return False
        return all(_ip_is_public(info[4][0]) for info in infos)
    except Exception:
        return False


# Backwards-compatible alias
_is_url_safe = is_safe_url


def _peer_ip(resp: requests.Response):
    """Best-effort address of the socket actually connected (closes the DNS-rebinding gap)."""
    try:
        return resp.raw._connection.sock.getpeername()[0]
    except Exception:
        return None


def safe_get(url: str, timeout: int = 8, max_bytes: int = MAX_DOWNLOAD_BYTES,
             session: requests.Session | None = None) -> tuple[requests.Response, bytes]:
    """
    GET with redirects followed manually so EVERY hop is validated, the connected
    peer is re-checked, and the body is size-capped. Raises ValueError if blocked.
    """
    sess = session or _get_session()
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not is_safe_url(current):
            raise ValueError(f"URL blocked for security reasons: {current}")
        resp = sess.get(current, timeout=timeout, allow_redirects=False, stream=True)
        try:
            peer = _peer_ip(resp)
            if peer and not _ip_is_public(peer):
                raise ValueError(f"URL blocked (resolved to non-public address): {current}")
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                if not loc:
                    raise ValueError("redirect without Location")
                current = urljoin(current, loc)
                continue
            resp.raise_for_status()
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ValueError(f"response too large ({declared} bytes)")
            body = bytearray()
            for chunk in resp.iter_content(chunk_size=16384):
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ValueError(f"response exceeded {max_bytes} bytes")
            return resp, bytes(body)
        finally:
            resp.close()
    raise ValueError("too many redirects")


def scrape_one(url : str, timeout: int = 8,max_chars: int =3000)-> str:
    """
    Scrape and clean one URL (SSRF-guarded, redirect-validated, size-capped).
    """
    resp, body = safe_get(url, timeout=timeout)
    ctype = resp.headers.get("Content-Type", "").lower()
    if ctype and not any(t in ctype for t in ("html", "text", "xml")):
        raise ValueError(f"unsupported content type: {ctype}")

    soup = BeautifulSoup(body, "lxml")
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        tag.extract()
    text = soup.get_text(separator = " ")
    clean_text= " ".join(text.split())
    return clean_text[:max_chars]



def scrape_many_parallel(urls: List[str], max_workers: int = 6) -> Dict[str, str]:
    """
    Parallel scrape URLs using ThreadPoolExecutor.
    Results are returned by URL key. Completion order is handled with as_completed.
    """
    if not urls:
        return {}

    results: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(scrape_one, url): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result(timeout=15)
            except Exception as exc:
                results[url] = f"ERROR scraping {url}: {exc}"

    return results

@tool
def parallel_scraper_tool(urls: List[str]) -> str:
    """
    Tool wrapper for parallel scraping.
    Returns labeled content by URL in one combined text blob.
    """
    data = scrape_many_parallel(urls=urls, max_workers=min(8, max(1, len(urls))))

    blocks = []
    for url, content in data.items():
        blocks.append(f"Source: {url}\n{content}")

    return "\n\n".join(blocks)