from numpy import deg2rad
import hashlib
import logging
import os
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from pipeline.fallback import execute_with_fallback, CascadeExecutionResult


load_dotenv()
log = logging.getLogger(__name__)

#env and model load 
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
VAULT_COLLECTION = "user_notes_vault_v1"
VECTOR_SIZE = 384
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


try : 
    _embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
except Exception as e :
    log.error(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}")
    _embedder = None 


async def _get_qdrant_client() -> AsyncQdrantClient : 
    return AsyncQdrantClient(url=QDRANT_URL , api_key=QDRANT_API_KEY)

async def ensure_vault_collection() :
    """Ensure the user notes collection and user_id index exist in Qdrant."""
    if not QDRANT_URL or not QDRANT_API_KEY :
        log.warning("QDRANT_URL or QDRANT_API_KEY is not set, skipping ensure_vault_collection")
        return
    client = await _get_qdrant_client()

    try : 
        exists = await client.collection_exists(collection_name=VAULT_COLLECTION)
        if not exists :
            await client.create_collection(
                collection_name=VAULT_COLLECTION,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                )
            )

            log.info("Created Qdrant Vault Collection '%s'", VAULT_COLLECTION)

        # Index user_id for fast user-isolated searches
        try:
            from qdrant_client.http import models as qdrant_models
            await client.create_payload_index(
                collection_name=VAULT_COLLECTION,
                field_name="userId",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
    finally:
        await client.close()


def _get_embedding(text : str)-> List[float] : 
    """Generate 384-d vector embedding using FastEmbed."""

    if _embedder is None :
        raise RuntimeError("Embedding model not loaded. Cannot generate embeddings.")

    embeddding = list(_embedder.embed([text]))
    return embeddding[0].tolist()


async def sync_note_to_vault(
    note_id: str,
    user_id: str,
    title: str,
    content: str,
    tags: Optional[List[str]] = None,
    source_url: Optional[str] = None,
) -> str:
    """
    Embeds note title + content and upserts into Qdrant.
    Returns: vector_point_id (embeddingId)
    """
    await ensure_vault_collection()
    combined_text = f"Title: {title}\nTags: {', '.join(tags or [])}\nContent: {content}"
    vector = _get_embedding(combined_text)
    # Deterministic UUID from note_id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}_{note_id}"))
    payload = {
        "noteId": note_id,
        "userId": user_id,
        "title": title,
        "content": content,
        "tags": tags or [],
        "sourceUrl": source_url or "",
    }
    client = await _get_qdrant_client()
    try:
        await client.upsert(
            collection_name=VAULT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        log.info("Synced note [%s] for user [%s] to vector vault.", note_id, user_id)
        return point_id
    finally:
        await client.close()


async def search_vault_notes(
    user_id: str,
    query: str,
    limit: int = 4,
) -> List[Dict]:
    """Semantic similarity search across a user's private notes."""
    await ensure_vault_collection()
    query_vector = _get_embedding(query)
    user_filter = Filter(
        must=[
            FieldCondition(
                key="userId",
                match=MatchValue(value=user_id),
            )
        ]
    )
    client = await _get_qdrant_client()
    try:
        # ✅ Qdrant-এর নতুন query_points API ব্যবহার
        results = await client.query_points(
            collection_name=VAULT_COLLECTION,
            query=query_vector,
            query_filter=user_filter,
            limit=limit,
        )
        notes = []
        for hit in results.points:
            note_data = hit.payload or {}
            note_data["score"] = hit.score
            notes.append(note_data)
        return notes
    finally:
        await client.close()



def draft_paper_section_with_notes(
    user_id: str,
    section_topic: str,
    instructions: str = "Draft a comprehensive, highly rigorous academic section.",
    retrieved_notes: Optional[List[Dict]] = None,
) -> CascadeExecutionResult:
    """
    RAG-powered Paper Section Drafter.
    Takes relevant research notes from user's vault and generates an academic section.
    """
    notes_context = ""
    if retrieved_notes:
        notes_context = "\n\n".join(
            f"--- Note: {n.get('title', 'Untitled')} ---\n{n.get('content', '')}"
            for n in retrieved_notes
        )
    else:
        notes_context = "No specific user notes attached. Synthesize from academic knowledge."
    system_prompt = (
        "You are an elite academic co-pilot for scientific paper writing. "
        "Your task is to draft a rigorous, peer-review quality paper section based on the user's "
        "provided research notes and instructions.\n\n"
        "Rules:\n"
        "1. Strictly ground your arguments in the provided Research Notes wherever possible.\n"
        "2. Use formal academic tone, clear section headings, and academic precision.\n"
        "3. Do NOT invent fake numerical data that contradicts the notes.\n"
        "4. Format the output in clean Markdown."
    )
    full_prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[USER RESEARCH NOTES CONTEXT]\n{notes_context}\n\n"
        f"[SECTION TOPIC]\n{section_topic}\n\n"
        f"[INSTRUCTIONS]\n{instructions}\n\n"
        f"[ACADEMIC SECTION DRAFT]"
    )
    log.info("Drafting paper section for topic: '%s'", section_topic)
    return execute_with_fallback(
        messages_or_prompt=full_prompt,
        custom_cascade=["worker-groq", "worker-gemini", "worker-mistral"]
    )
