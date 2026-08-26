"""
api/server.py
────────────────────────────────────────────────────────────
Production FastAPI server with Async Job Queue pattern.

WHY JOB QUEUE?
  The research pipeline takes 90-120 seconds.
  HTTP timeout on most platforms = 30-60s.
  Solution: POST /research → returns {job_id} immediately,
            client polls GET /job/{id} until done.

FLOW:
  1. POST /research {"topic":"..."}
        → creates job in Redis
        → starts background thread
        → returns {"job_id": "abc123", "status": "queued"}

  2. GET /job/{job_id}
        → returns {status: "running"|"done"|"failed", progress: X%, result: ...}

  3. GET /stream/{job_id}   (Server-Sent Events)
        → streams progress events as they happen

Run locally:
  uvicorn api.server:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional

# ── Ensure project root is always on sys.path ──────────────────────────────────
# Works whether you run:  python api/server.py   OR   uvicorn api.server:app
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Now safe to import project modules ────────────────────────────────────────
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Research System",
    description="AI-powered deep research: QueryRewrite → Search → RAG → FactCheck → Report",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For dev only - no credentials support
    allow_credentials=False,  # Set True only with explicit origins
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Job Store (Upstash Redis or in-memory fallback) ───────────────────────────

class JobStore:
    """
    Stores job state in Upstash Redis.
    Falls back to a plain dict if Redis is unavailable (dev mode).
    """
    _local: dict = {}   # fallback in-memory store
    _locks: dict = {}   # per-job locks
    _global_lock = threading.Lock()

    def _get_lock(self, job_id: str) -> threading.Lock:
        with self._global_lock:
            if job_id not in self._locks:
                self._locks[job_id] = threading.Lock()
            return self._locks[job_id]

    def _redis_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def set(self, job_id: str, data: dict, ttl: int = 3600) -> None:
        from tools.cache_tool import cache_put
        try:
            cache_put(self._redis_key(job_id), data, ttl=ttl)
        except Exception:
            self._local[job_id] = data  # fallback

    def get(self, job_id: str) -> Optional[dict]:
        from tools.cache_tool import cache_get
        try:
            result = cache_get(self._redis_key(job_id))
            if result is not None:
                return result
        except Exception:
            pass
        return self._local.get(job_id)  # fallback

    def update(self, job_id: str, patch: dict) -> None:
        with self._get_lock(job_id):
            existing = self.get(job_id) or {}
            existing.update(patch)
            self.set(job_id, existing)


_jobs = JobStore()


# ── Models ─────────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    user_id: Optional[str] = None
    mode: str = Field(default="deep", pattern=r"^(fast|deep)$")

class SaveHistoryRequest(BaseModel):
    job_id: str
    user_id: str
    topic: str
    report: str
    critique: str
    score: float
    fact_score: float
    urls: list
    time_sec: float


class JobResponse(BaseModel):
    job_id: str
    status: str          # queued | running | done | failed
    progress: int = 0    # 0–100
    stage: str = ""      # current pipeline stage name
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = 0.0


# ── Background runner ──────────────────────────────────────────────────────────




def _run_pipeline_background(job_id: str, topic: str, user_id: Optional[str] = None, mode: str = "deep") -> None:
    """
    Runs in a daemon thread.
    Updates job store at each stage so the polling endpoint reflects progress.
    """
    try:
        # ── Patch pipeline to emit progress updates ────────────────────────
        from pipeline import stream_research as run_research_stream

        _jobs.update(job_id, {"status": "running", "progress": 5, "stage": "starting"})

        final_result = {}

        # Capture real events from LangGraph
        for event in run_research_stream(topic, job_id=job_id, mode=mode):
            if "__done__" in event:
                final_result = event["__done__"]
                continue
            node_name = list(event.keys())[0]

            if node_name == "planner":
                _jobs.update(job_id, {"stage": " Planning Research...", "progress": 10})

            elif node_name == "searcher":
                _jobs.update(job_id, {"stage": " Searching the web...", "progress": 30})

            elif node_name == "reader":
                _jobs.update(job_id, {"stage": " Reading articles...", "progress": 45})

            elif node_name == "summarize":
                _jobs.update(job_id, {"stage": " Summarising content...", "progress": 55})

            elif node_name == "rag":
                _jobs.update(job_id, {"stage": " RAG Retrieval...", "progress": 65})

            elif node_name == "writer":
                _jobs.update(job_id, {"stage": " Writing report...", "progress": 80})

            elif node_name == "fact_check":
                _jobs.update(job_id, {"stage": " Fact-checking...", "progress": 90})

            elif node_name == "critic":
                _jobs.update(job_id, {"stage": " Reviewing quality...", "progress": 95})

            elif node_name == "knowledge_graph":
                _jobs.update(job_id, {"stage": " Building Knowledge Graph...", "progress": 98})

            elif node_name == "prepare_rewrite":
                _jobs.update(job_id, {"stage": " Revising report based on feedback...", "progress": 75})

            elif node_name == "__done__":
                final_result = event["__done__"]

        # Job finished
        job_result = {
            "job_id":           job_id,
            "user_id":          user_id,
            "topic":            final_result.get("topic", topic),
            "report":           final_result.get("report", ""),
            "critique":         final_result.get("critique", ""),
            "critique_score":   final_result.get("critique_score", 0),
            "fact_check_score": final_result.get("fact_check_score", 0.0),
            "rewritten_queries": final_result.get("rewritten_queries", []),
            "verified_urls":    final_result.get("verified_urls", []),
            "time_sec":         final_result.get("time_sec", 0),
            "error":            final_result.get("error", ""),
        }
        _jobs.update(job_id, {
            "status":   "done",
            "progress": 100,
            "stage":    " Complete",
            "result":   job_result,
        })

        # Persist to Supabase history (best effort)
        try:
            from memory.history import save_research
            save_research(job_result)
        except Exception as db_exc:
            log.warning("Failed to save completed job %s to Supabase history: %s", job_id, db_exc)

        log.info("Job %s done in %.1fs", job_id, final_result.get("time_sec", 0))

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        _jobs.update(job_id, {
            "status":   "failed",
            "progress": 0,
            "stage":    " Failed",
            "error":    str(exc),
        })
# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — API overview."""
    return {
        "service": "Multi-Agent Research System",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "start_research": "POST /research",
            "poll_job":       "GET  /job/{job_id}",
            "stream_job":     "GET  /stream/{job_id}",
            "history":        "GET  /history",
            "health":         "GET  /health",
        },
    }


@app.get("/health")
async def health():
    """Health check — returns 200 if the server is alive."""
    return {"status": "ok", "service": "multi-agent-research", "version": "3.0.0"}



@app.post("/research", response_model=JobResponse, status_code=202)
async def start_research(req: ResearchRequest):
    """
    Start a research job.

    Returns immediately with a job_id.
    Poll GET /job/{job_id} for status and result.
    """
    job_id = str(uuid.uuid4())[:12]
    now    = time.time()

    job_data = {
        "job_id":     job_id,
        "topic":      req.topic,
        "status":     "queued",
        "progress":   0,
        "stage":      " Queued",
        "result":     None,
        "error":      None,
        "created_at": now,
    }
    _jobs.set(job_id, job_data, ttl=7200)  # keep for 2h

    # Fire background thread
    t = threading.Thread(
        target=_run_pipeline_background,
        args=(job_id, req.topic, req.user_id, req.mode),
        daemon=True,
    )
    t.start()

    log.info("Job %s queued for topic=%r user_id=%r", job_id, req.topic, req.user_id)
    return job_data


@app.get("/job/{job_id}", response_model=JobResponse)
async def poll_job(job_id: str):
    """
    Poll job status.

    status values:
      queued  — waiting to start
      running — pipeline is executing
      done    — result ready in 'result' field
      failed  — error in 'error' field
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    """
    Server-Sent Events (SSE) stream for a job.

    Emits progress events every 2 seconds until job is done or failed.
    The React frontend subscribes to this for real-time updates.
    """
    async def event_generator():
        for _ in range(300):   # max 600s (300 × 2s)
            job = _jobs.get(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'message': 'job not found'})}\n\n"
                return

            import asyncio
            yield f"event: progress\ndata: {json.dumps(job)}\n\n"

            if job["status"] in ("done", "failed"):
                return

            await asyncio.sleep(2)

        yield f"event: timeout\ndata: {{}}\n\n"

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history")
async def get_history(limit: int = 10, userId: Optional[str] = None, user_id: Optional[str] = None):
    """List recent completed research sessions from Supabase."""
    import asyncio
    try:
        from memory import get_recent
        target_uid = user_id or userId
        log.info("=== Python API /history received userId=%r, user_id=%r, using target_uid=%r ===", userId, user_id, target_uid)
        loop = asyncio.get_event_loop()
        records = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: get_recent(limit=limit, user_id=target_uid)),
            timeout=8.0,
        )
        return {"records": records, "count": len(records)}
    except asyncio.TimeoutError:
        return {"records": [], "count": 0, "note": "DB unavailable (timeout)"}
    except Exception as exc:
        log.warning("History endpoint failed: %s", exc)
        return {"records": [], "count": 0, "note": str(exc)}


@app.post("/history")
async def save_history_record(req: SaveHistoryRequest):
    """Save a history record directly to Supabase."""
    try:
        from memory.history import save_research
        job_result = {
            "job_id": req.job_id,
            "user_id": req.user_id,
            "topic": req.topic,
            "report": req.report,
            "critique": req.critique,
            "critique_score": req.score,
            "fact_check_score": req.fact_score,
            "rewritten_queries": [],
            "verified_urls": req.urls,
            "time_sec": req.time_sec,
            "error": "",
        }
        loop = asyncio.get_event_loop()
        rec_id = await loop.run_in_executor(None, lambda: save_research(job_result))
        return {"success": True, "id": rec_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/{record_id}")
async def get_history_item(record_id: str):
    """Get a single research record by numeric ID or job_id."""
    try:
        from memory.history import get_by_id
        loop = asyncio.get_event_loop()
        record = await loop.run_in_executor(None, get_by_id, record_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        return record
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/cache/stats")
async def get_cache_stats():
    """Upstash Redis cache statistics."""
    try:
        from tools.cache_tool import cache_stats
        return cache_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



# ── Phase 2: Paraphraser Endpoint ─────────────────────────────────────────────

class ParaphraseRequest(BaseModel) : 
    text : str = Field(..., min_length=10, max_length=5000)
    mode : str = Field(
        default="academic",
        pattern=r"^(academic|simplify|executive|humanize)$",
        
    )
    user_id : Optional[str] = None

class ParaphraseResponse(BaseModel) :
    paraphrased_text: str
    mode: str
    provider_used: str
    duration_sec: float
    token_usage: dict

@app.post("/api/v1/paraphrase", response_model=ParaphraseResponse, tags=["Phase 2: Paper Studio"])
def paraphrase_text(req: ParaphraseRequest):
    """
    AI Paraphraser — 4 modes:
      • academic   → Journal-quality formal rewrite
      • simplify   → ELI15 plain language
      • executive  → 3-bullet point summary
      • humanize   → Remove AI-sounding patterns
    """
    try:
        from agents.paraphraser_agent import paraphrase
        result = paraphrase(text=req.text, mode=req.mode)
        return ParaphraseResponse(
            paraphrased_text=result.content,
            mode=req.mode,
            provider_used=result.provider_used,
            duration_sec=result.duration_sec,
            token_usage=result.token_usage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"All AI providers failed: {exc}")
    except Exception as exc:
        log.exception("Unexpected error in /paraphrase")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Use app object directly (not string) so it works when run as:
    #   python api/server.py  (from project root)
    # For hot-reload use uvicorn CLI instead:
    #   uvicorn api.server:app --reload   (from project root)
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,    # reload=True requires running via uvicorn CLI
        log_level="info",
    )


