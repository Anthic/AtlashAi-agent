# Migrated: Mistral Embeddings -> FastEmbed (BAAI/bge-small-en-v1.5)
"""
pipeline/rag.py
Per-run retrieval over the scraped sources.

Fixes vs. the previous version (AI-9):
  * Chunks are keyed by a hash of (topic + content), NOT the raw topic, so an
    existing topic never causes new content to be skipped and old/other-user
    chunks are never returned.
  * Indexes the RAW scraped pages (summarised text carries little extra signal).
  * One shared sync Qdrant client; collection/index setup runs once per process
    instead of 3 extra calls on every job.
  * Old chunks expire (RAG_TTL_DAYS) so the shared collection does not grow forever.
"""

import hashlib
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.models import VectorParams, Distance

load_dotenv()

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

QDRANT_URL      = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
MODEL_NAME      = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE     = 384
COLLECTION_NAME = "research_bge_v1"
RAG_TTL_DAYS    = int(os.getenv("RAG_TTL_DAYS", "7"))
MAX_RAG_CHARS   = 40000

# ── Model Singleton ────────────────────────────────────────────────────────────

log.info("RAG: Loading FastEmbed model '%s'...", MODEL_NAME)
try:
    _embeddings_model = TextEmbedding(model_name=MODEL_NAME)
    log.info("FastEmbed model loaded, vector_size=384")
except Exception as e:
    log.error("Failed to load FastEmbed model: %s", e)
    _embeddings_model = None

# ── Shared client + one-time setup ─────────────────────────────────────────────

_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()
_ready = False
_last_cleanup = 0.0


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not QDRANT_URL or not QDRANT_API_KEY:
                    raise EnvironmentError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
                _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
    return _client


def _ensure_ready(client: QdrantClient) -> None:
    """Create the collection + payload indexes once per process."""
    global _ready
    if _ready:
        return
    with _client_lock:
        if _ready:
            return
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            log.info("RAG: Created Qdrant collection '%s'", COLLECTION_NAME)
        for field, schema in (
            ("metadata.run_key", qm.PayloadSchemaType.KEYWORD),
            ("metadata.created_at", qm.PayloadSchemaType.INTEGER),
        ):
            try:
                client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
            except Exception as e:
                log.debug("RAG: payload index %s: %s", field, e)
        _ready = True


def _cleanup_old(client: QdrantClient) -> None:
    """Best-effort: drop chunks older than RAG_TTL_DAYS (and legacy chunks with no timestamp). Hourly at most."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < 3600:
        return
    _last_cleanup = now
    try:
        cutoff = int(now - RAG_TTL_DAYS * 86400)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qm.FilterSelector(filter=qm.Filter(should=[
                qm.FieldCondition(key="metadata.created_at", range=qm.Range(lt=cutoff)),
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key="metadata.created_at")),
            ])),
        )
    except Exception as e:
        log.debug("RAG: cleanup skipped: %s", e)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk_text(raw_text: str) -> List[Document]:
    """Split raw scraped text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    return splitter.create_documents([raw_text])


def run_key_for(topic: str, content: str) -> str:
    """Isolation key: same topic + same content => same key (idempotent re-ingest);
    different content or topic => different key (nothing stale or shared leaks in)."""
    h = hashlib.sha256()
    h.update(topic.lower().strip().encode("utf-8"))
    h.update(b"\x00")
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:32]


def _embed(texts: List[str]) -> List[List[float]]:
    if not _embeddings_model:
        raise ValueError("FastEmbed model is not initialized.")
    return [[float(x) for x in v] for v in _embeddings_model.embed(texts)]


# ── RAG execution ──────────────────────────────────────────────────────────────

def _run_rag(state: Dict) -> Dict:
    topic = state.get("topic", "").strip()

    # Prefer the RAW scraped pages: retrieval over already-summarised text adds little.
    raw_text = (
        state.get("scraped_content", "").strip()
        or state.get("summarized_content", "").strip()
    )[:MAX_RAG_CHARS]

    if not raw_text:
        return {
            **state,
            "rag_context": "",
            "error": state.get("error", "No content available for RAG."),
        }

    try:
        client = _get_client()
        _ensure_ready(client)
        _cleanup_old(client)

        run_key = run_key_for(topic, raw_text)
        run_filter = qm.Filter(must=[
            qm.FieldCondition(key="metadata.run_key", match=qm.MatchValue(value=run_key))
        ])

        try:
            already = client.count(COLLECTION_NAME, count_filter=run_filter, exact=False).count > 0
        except Exception:
            already = False

        if not already:
            chunk_texts = [c.page_content for c in _chunk_text(raw_text)]
            if chunk_texts:
                embeddings = _embed(chunk_texts)
                created = int(time.time())
                points = [
                    qm.PointStruct(
                        id=hashlib.md5(f"{run_key}-{i}".encode()).hexdigest(),
                        vector=emb,
                        payload={
                            "page_content": chunk,
                            "metadata": {"topic": topic, "run_key": run_key, "created_at": created},
                        },
                    )
                    for i, (chunk, emb) in enumerate(zip(chunk_texts, embeddings))
                ]
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                log.info("RAG: ingested %d chunks (run_key=%s)", len(points), run_key[:8])
        else:
            log.info("RAG: content already indexed (run_key=%s), skipping ingest", run_key[:8])

        query_vector = _embed([topic])[0]
        docs = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=run_filter,
            limit=4,
        ).points

        rag_context = "\n\n".join(
            f"[Chunk {i+1}]: {doc.payload.get('page_content', '')}"
            for i, doc in enumerate(docs)
        )
        log.info("RAG: retrieved %d chunks (%d chars) for topic=%r", len(docs), len(rag_context), topic)
        return {**state, "rag_context": rag_context}

    except Exception as exc:
        log.exception("RAG node failed")
        return {**state, "rag_context": "", "error": f"RAG failed: {exc}"}


def run_rag_node(state: Dict) -> Dict:
    """
    LangGraph node — embeds the scraped content into Qdrant (keyed per run) and
    retrieves the most relevant chunks for the current topic. Synchronous: it runs
    in a worker thread, so no event loop is created per job.
    """
    return _run_rag(state)
