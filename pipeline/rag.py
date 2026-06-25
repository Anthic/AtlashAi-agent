# Migrated: Mistral Embeddings -> FastEmbed (BAAI/bge-small-en-v1.5)

import hashlib
import logging
import os
import asyncio
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

QDRANT_URL      = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
MODEL_NAME      = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE     = 384
COLLECTION_NAME = "research_bge_v1"

# ── Model Singleton ────────────────────────────────────────────────────────────

log.info("RAG: Loading FastEmbed model '%s'...", MODEL_NAME)
try:
    _embeddings_model = TextEmbedding(model_name=MODEL_NAME)
    log.info("FastEmbed model loaded, vector_size=384")
except Exception as e:
    log.error("Failed to load FastEmbed model: %s", e)
    _embeddings_model = None


# ── Collection Startup Check ──────────────────────────────────────────────────

async def _async_init_collection(client: AsyncQdrantClient = None):
    """Ensure the Qdrant collection and payload indexes exist."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        log.warning("RAG: QDRANT_URL or QDRANT_API_KEY not configured. Startup collection check skipped.")
        return

    should_close = False
    if client is None:
        client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        should_close = True

    try:
        exists = await client.collection_exists(collection_name=COLLECTION_NAME)
        if not exists:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            log.info("RAG: Created Qdrant collection '%s' with size=%d", COLLECTION_NAME, VECTOR_SIZE)
        else:
            log.info("RAG: Qdrant collection '%s' already exists", COLLECTION_NAME)

        # Create payload index for topic metadata (strict requirement in Qdrant for filtering)
        try:
            from qdrant_client.http import models as qdrant_models
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.topic",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD
            )
            log.info("RAG: Verified payload index for 'metadata.topic' exists")
        except Exception as e:
            log.warning("RAG: Could not ensure payload index for 'metadata.topic': %s", e)

    except Exception as e:
        log.error("RAG startup check failed for collection '%s': %s", COLLECTION_NAME, e)
    finally:
        if should_close:
            await client.close()


# Run startup check in a non-blocking background thread
def _start_init_thread():
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_async_init_collection())
        loop.close()
    except Exception as e:
        log.warning("RAG: Failed to complete startup collection initialization: %s", e)

import threading
threading.Thread(target=_start_init_thread, daemon=True).start()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk_text(raw_text: str) -> List[Document]:
    """Split raw scraped text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    return splitter.create_documents([raw_text])


# ── Async RAG Execution ────────────────────────────────────────────────────────

async def _async_run_rag(state: Dict) -> Dict:
    topic = state.get("topic", "").strip()

    # Prefer summarized content (less noise)
    raw_text = (
        state.get("summarized_content", "").strip()
        or state.get("scraped_content", "").strip()
    )

    if not raw_text:
        return {
            **state,
            "rag_context": "",
            "error": state.get("error", "No content available for RAG."),
        }

    try:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise EnvironmentError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

        async_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Guarantee that the collection and payload indexes exist before performing any operations
        await _async_init_collection(async_client)
        
        # Check if topic already exists in COLLECTION_NAME to avoid duplicate ingestion
        from qdrant_client.http import models as qdrant_models
        
        filter_topic = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.topic",
                    match=qdrant_models.MatchValue(value=topic)
                )
            ]
        )
        
        # Check if vectors already ingested for this topic
        is_new = True
        try:
            count_res = await async_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=filter_topic
            )
            is_new = count_res.count == 0
        except Exception:
            is_new = True

        chunks = _chunk_text(raw_text)
        chunk_texts = [c.page_content for c in chunks]

        loop = asyncio.get_event_loop()

        async def embed_chunks_task(texts):
            if not _embeddings_model:
                raise ValueError("FastEmbed model is not initialized.")
            # fastembed is CPU-bound, run in executor to prevent blocking the event loop
            return await loop.run_in_executor(None, lambda: [list(v) for v in _embeddings_model.embed(texts)])

        async def embed_query_task(q_text):
            if not _embeddings_model:
                raise ValueError("FastEmbed model is not initialized.")
            # fastembed is CPU-bound, run in executor to prevent blocking the event loop
            return await loop.run_in_executor(None, lambda: list(next(_embeddings_model.embed([q_text]))))

        # Parallel gather embedding & query embedding
        if is_new and chunk_texts:
            chunk_embeddings, query_vector = await asyncio.gather(
                embed_chunks_task(chunk_texts),
                embed_query_task(topic)
            )
            
            # Prepare points for ingestion
            points = [
                qdrant_models.PointStruct(
                    id=hashlib.md5(f"{topic}-{i}-{chunk[:20]}".encode()).hexdigest(),
                    vector=emb,
                    payload={"page_content": chunk, "metadata": {"topic": topic}}
                )
                for i, (chunk, emb) in enumerate(zip(chunk_texts, chunk_embeddings))
            ]
            
            try:
                await async_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                log.info("RAG: ingested %d chunks into '%s'", len(points), COLLECTION_NAME)
            except Exception as e:
                log.error("RAG: failed to upsert points: %s", e)
        else:
            log.info("RAG: collection '%s' already has data for topic=%r, skipping ingest", COLLECTION_NAME, topic)
            query_vector = await embed_query_task(topic)

        # Similarity search
        docs = []
        try:
            search_res = await async_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=filter_topic,
                limit=4
            )
            docs = search_res.points
        except Exception as e:
            log.error("RAG: similarity search failed: %s", e)

        await async_client.close()

        rag_context = "\n\n".join(
            f"[Chunk {i+1}]: {doc.payload.get('page_content', '')}"
            for i, doc in enumerate(docs)
        )

        log.info(
            "RAG: retrieved %d chunks (%d chars) for topic=%r",
            len(docs), len(rag_context), topic,
        )
        return {**state, "rag_context": rag_context}

    except Exception as exc:
        log.exception("RAG node failed")
        return {**state, "rag_context": "", "error": f"RAG failed: {exc}"}


# ── Main Node Signature (Synchronous wrapper) ───────────────────────────
# ── Main Node Signature (Synchronous wrapper) ───────────────────────────

def run_rag_node(state: Dict) -> Dict:
    """
    LangGraph node — embeds scraped content into Qdrant and retrieves
    the most relevant chunks for the current topic.

    Uses BAAI/bge-small-en-v1.5 locally via fastembed and gathers
    chunk + query embedding in parallel.
    """
    return asyncio.run(_async_run_rag(state))