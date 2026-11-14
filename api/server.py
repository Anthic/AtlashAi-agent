

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
import secrets

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

log = logging.getLogger(__name__)

# ── Security config ───────────────────────────────────────────────────────────
# Shared secret between the Node backend and this service. When set, every route
# except "/" and "/health" requires header  X-Internal-Key: <secret>.
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
_PUBLIC_PATHS = {"/", "/health"}

# Browser origins allowed to call this API directly. Empty (default) = none: the
# Node backend calls server-to-server and needs no CORS. Never "*".
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# A finished job whose report is shorter than this is treated as failed.
MIN_REPORT_CHARS = int(os.getenv("MIN_REPORT_CHARS", "200"))

# Input size caps (one request must not push megabytes into the LLM cascade).
MAX_DOC_CHARS = int(os.getenv("MAX_DOC_CHARS", "60000"))
MAX_NOTE_CHARS = int(os.getenv("MAX_NOTE_CHARS", "50000"))
MAX_ANSWER_CHARS = 10000


def require_internal_key(request: Request, x_internal_key: str = Header(default="")):
    """Reject callers that do not present the shared secret (constant-time compare)."""
    if request.url.path in _PUBLIC_PATHS or not INTERNAL_KEY:
        return
    if not secrets.compare_digest(x_internal_key.encode(), INTERNAL_KEY.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


if not INTERNAL_KEY:
    log.warning(
        "INTERNAL_API_KEY is NOT set: the Python API accepts unauthenticated requests. "
        "Set INTERNAL_API_KEY here and in the Node backend (X-Internal-Key header)."
    )

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Research System",
    description="AI-powered deep research: QueryRewrite → Search → RAG → FactCheck → Report",
    version="3.0.0",
    dependencies=[Depends(require_internal_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Internal-Key"],
)


# ── Job Store (Upstash Redis or in-memory fallback) ───────────────────────────

class JobStore:
    """
    Job state store.

    The in-process dict is the source of truth for jobs this process runs (the
    pipeline thread lives here), and Upstash Redis is a best-effort mirror so other
    workers / restarts can still see the job. Reading local-first also removes
    ~1 Redis command per poll, which is what exhausted the free quota
    (10K commands/day) and made every poll return "job not found".
    """
    _local: dict = {}
    _local_ts: dict = {}
    _locks: dict = {}
    _global_lock = threading.Lock()
    _LOCAL_TTL = 7200

    def _get_lock(self, job_id: str) -> threading.Lock:
        with self._global_lock:
            if job_id not in self._locks:
                self._locks[job_id] = threading.Lock()
            return self._locks[job_id]

    def _redis_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def _purge_expired(self) -> None:
        cutoff = time.time() - self._LOCAL_TTL
        for jid in [j for j, t in self._local_ts.items() if t < cutoff]:
            self._local.pop(jid, None)
            self._local_ts.pop(jid, None)
            self._locks.pop(jid, None)

    def set(self, job_id: str, data: dict, ttl: int = 3600, mirror: bool = True) -> None:
        with self._global_lock:
            self._local[job_id] = data
            self._local_ts[job_id] = time.time()
            self._purge_expired()
        if mirror:
            from tools.cache_tool import cache_put
            if not cache_put(self._redis_key(job_id), data, ttl=ttl):
                log.debug("JobStore: Redis mirror skipped for %s (using in-memory copy)", job_id)

    def get(self, job_id: str) -> Optional[dict]:
        local = self._local.get(job_id)
        if local is not None:
            return local
        from tools.cache_tool import cache_get
        try:
            return cache_get(self._redis_key(job_id))
        except Exception:
            return None

    def update(self, job_id: str, patch: dict) -> None:
        with self._get_lock(job_id):
            existing = dict(self.get(job_id) or {})
            existing.update(patch)
            # Mirror to Redis only on meaningful transitions (not every progress tick)
            terminal = existing.get("status") in ("done", "failed")
            self.set(job_id, existing, ttl=7200, mirror=terminal or "status" in patch)


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
    report: str = Field(..., max_length=MAX_DOC_CHARS * 2)
    critique: str = Field(default="", max_length=20000)
    score: float
    fact_score: float
    urls: list = Field(default_factory=list, max_length=100)
    time_sec: float


class JobResponse(BaseModel):
    job_id: str
    # Returned to the trusted API gateway so it can enforce ownership before
    # sending the job state to a browser.
    user_id: Optional[str] = None
    status: str          # queued | running | done | failed
    progress: int = 0    # 0–100
    stage: str = ""      # current pipeline stage name
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = 0.0

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



class VaultSyncRequest(BaseModel):
    note_id: str
    user_id: str
    title: str = Field(..., max_length=500)
    content: str = Field(..., max_length=MAX_NOTE_CHARS)
    tags: Optional[list] = Field(default_factory=list, max_length=50)
    source_url: Optional[str] = None
class DraftSectionRequest(BaseModel):
    user_id: str
    section_topic: str = Field(..., min_length=3, max_length=1000)
    instructions: Optional[str] = Field(default="Draft a comprehensive academic section.", max_length=2000)


class GapFinderRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=1000)
    literature_context: Optional[str] = Field(default=None, max_length=MAX_DOC_CHARS)
    user_id: Optional[str] = None

class PeerReviewRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=500)
    content: str = Field(..., min_length=20, max_length=MAX_DOC_CHARS)
    user_id: Optional[str] = None

class SlideDeckRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=500)
    content: Optional[str] = Field(default=None, max_length=MAX_DOC_CHARS)
    num_slides: Optional[int] = Field(default=8, ge=5, le=15)
    user_id: Optional[str] = None

class DefenseQuestionsRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=500)
    content: str = Field(..., min_length=20, max_length=MAX_DOC_CHARS)
    user_id: Optional[str] = None

class DefenseEvaluateRequest(BaseModel):
    examiner_name: str = Field(..., max_length=200)
    examiner_title: str = Field(..., max_length=200)
    question: str = Field(..., max_length=3000)
    student_answer: str = Field(..., min_length=5, max_length=MAX_ANSWER_CHARS)
    user_id: Optional[str] = None

class TemporalFactCheckRequest(BaseModel):
    title: str = Field(..., max_length=500)
    content: str = Field(..., max_length=MAX_DOC_CHARS)
    user_id: Optional[str] = None
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
        report = (final_result.get("report") or "").strip()
        if len(report) < MIN_REPORT_CHARS:
            # Do not present an empty / error-only run as "done": the Node side treats
            # "failed" as the trigger to refund the charge or free-quota use.
            reason = final_result.get("error") or "empty_report"
            log.error("Job %s produced no usable report (%d chars): %s", job_id, len(report), reason)
            _jobs.update(job_id, {
                "status":   "failed",
                "progress": 0,
                "stage":    " Failed",
                "error":    f"empty_report: {reason}"[:500],
            })
            try:
                from pipeline.alert_watcher import notify_pipeline_error
                notify_pipeline_error(stage="Research produced empty report", topic=topic,
                                      error_msg=str(reason), user_id=user_id or "Anonymous")
            except Exception:
                pass
            return

        job_result = {
            "job_id":           job_id,
            "user_id":          user_id,
            "topic":            final_result.get("topic", topic),
            "report":           report,
            "critique":         final_result.get("critique", ""),
            "critique_score":   final_result.get("critique_score", 0),
            "fact_check_score": final_result.get("fact_check_score", 0.0),
            "fact_check_parsed": final_result.get("fact_check_parsed", False),
            "knowledge_graph":  final_result.get("knowledge_graph") or {},      # was dropped before
            "knowledge_graph_md": final_result.get("knowledge_graph_md", ""),
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
        try:
            from pipeline.alert_watcher import notify_pipeline_error
            notify_pipeline_error(
                stage="Deep Research Pipeline Execution",
                topic=topic,
                error_msg=str(exc),
                user_id=user_id or "Anonymous",
            )
        except Exception:
            pass
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
    return {
        "status": "ok",
        "service": "multi-agent-research",
        "version": "3.0.0",
        "auth": "enabled" if INTERNAL_KEY else "DISABLED",
    }



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
        "user_id":    req.user_id,
        "topic":      req.topic,
        "status":     "queued",
        "progress":   0,
        "stage":      " Queued",
        "result":     None,
        "error":      None,
        "created_at": now,
    }
    await run_in_threadpool(_jobs.set, job_id, job_data, 7200)  # keep for 2h

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
    job = await run_in_threadpool(_jobs.get, job_id)
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
            job = await run_in_threadpool(_jobs.get, job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'message': 'job not found'})}\n\n"
                return

            yield f"event: progress\ndata: {json.dumps(job)}\n\n"

            if job["status"] in ("done", "failed"):
                return

            await asyncio.sleep(2)

        yield f"event: timeout\ndata: {{}}\n\n"

    job = await run_in_threadpool(_jobs.get, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history")
async def get_history(limit: int = 10, userId: Optional[str] = None, user_id: Optional[str] = None):
    """List recent completed research sessions from Supabase."""
    try:
        from memory import get_recent
        target_uid = user_id or userId
        log.info("Python API /history userId=%r user_id=%r -> %r", userId, user_id, target_uid)
        loop = asyncio.get_running_loop()
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
        rec_id = await run_in_threadpool(save_research, job_result)
        return {"success": True, "id": rec_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/{record_id}")
async def get_history_item(record_id: str, user_id: Optional[str] = None):
    """Get a single research record by numeric ID or job_id.

    Pass ?user_id=<verified user> so ownership is enforced (ids are sequential and
    guessable). Without it the lookup is unscoped and only trusted callers
    (holding X-Internal-Key) can reach it.
    """
    try:
        from memory.history import get_by_id
        if not user_id:
            log.warning("GET /history/%s called without user_id — unscoped lookup", record_id)
        record = await run_in_threadpool(get_by_id, record_id, user_id)
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
        return await run_in_threadpool(cache_stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



# ── Phase 2: Paraphraser Endpoint ─────────────────────────────────────────────



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

# ── Phase 2: Smart Notes Vault & Section Drafter ───────────────────────────────


@app.post("/api/v1/vault/sync", tags=["Phase 2: Paper Studio"])
async def sync_note(req: VaultSyncRequest):
    """Sync a note to Qdrant vector vault for RAG."""
    try:
        from agents.note_vault_agent import sync_note_to_vault
        point_id = await sync_note_to_vault(
            note_id=req.note_id,
            user_id=req.user_id,
            title=req.title,
            content=req.content,
            tags=req.tags,
            source_url=req.source_url,
        )
        return {"success": True, "embedding_id": point_id}
    except Exception as exc:
        log.exception("Error syncing note to vault")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/v1/vault/notes/{note_id}", tags=["Phase 2: Paper Studio"])
async def delete_note(note_id: str, user_id: str):
    """Remove a note's vector from the vault (call after a note is deleted, or before re-sync on edit)."""
    try:
        from agents.note_vault_agent import delete_note_vectors
        await delete_note_vectors(note_id=note_id, user_id=user_id)
        return {"success": True}
    except Exception as exc:
        log.exception("Error deleting note vectors")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/vault/draft-section", tags=["Phase 2: Paper Studio"])
async def draft_section(req: DraftSectionRequest):
    """RAG-powered paper section drafter using user's saved notes."""
    try:
        from agents.note_vault_agent import search_vault_notes, draft_paper_section_with_notes
        notes = await search_vault_notes(user_id=req.user_id, query=req.section_topic, limit=4)
        result = await run_in_threadpool(
            draft_paper_section_with_notes,
            user_id=req.user_id,
            section_topic=req.section_topic,
            instructions=req.instructions,
            retrieved_notes=notes,
        )
        return {
            "draft_markdown": result.content,
            "notes_referenced": len(notes),
            "provider_used": result.provider_used,
            "duration_sec": result.duration_sec,
            "token_usage": result.token_usage,
        }
    except Exception as exc:
        log.exception("Error drafting paper section")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Phase 3: Literature Matrix & Gap Finder ────────────────────────────────────


@app.post("/api/v1/academic/gap-finder", tags=["Phase 3: Academic AI"])
def find_research_gaps(req: GapFinderRequest):
    """
    Analyzes a topic/literature to construct:
      1. Literature Comparison Matrix (Markdown Table)
      2. Unresolved Research Gaps
      3. Novel Research Proposals
    """
    try:
        from agents.gap_finder_agent import analyze_research_gaps
        result = analyze_research_gaps(
            topic=req.topic,
            literature_context=req.literature_context,
        )
        return {
            "topic": req.topic,
            "gap_analysis_markdown": result.content,
            "provider_used": result.provider_used,
            "duration_sec": result.duration_sec,
            "token_usage": result.token_usage,
        }
    except Exception as exc:
        log.exception("Error analyzing research gaps")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Phase 3: Simulated 3-Agent Peer-Review Panel ───────────────────────────────


@app.post("/api/v1/academic/peer-review", tags=["Phase 3: Academic AI"])
def peer_review_paper(req: PeerReviewRequest):
    """
    Simulates a 3-agent peer review panel:
      • Reviewer 1: Methodology Critic
      • Reviewer 2: Domain & Novelty Scholar
      • Reviewer 3: Clarity & Overclaiming Auditor
    """
    try:
        from agents.peer_review_agent import review_paper_draft
        result = review_paper_draft(
            paper_title=req.title,
            paper_content=req.content,
        )
        return {
            "title": req.title,
            "review_markdown": result.content,
            "provider_used": result.provider_used,
            "duration_sec": result.duration_sec,
            "token_usage": result.token_usage,
        }
    except Exception as exc:
        log.exception("Error running peer review")
        raise HTTPException(status_code=500, detail=str(exc))



@app.post("/api/v1/academic/generate-slides", tags=["Phase 3: Academic AI"])
def create_slide_deck(req: SlideDeckRequest):
    """
    Transforms research text or topic into an 8-10 slide Marp Markdown Presentation.
    """
    try:
        from agents.slide_generator_agent import generate_slide_deck
        result = generate_slide_deck(
            title_or_topic=req.title,
            content=req.content,
            num_slides=req.num_slides or 8,
        )
        return {
            "title": req.title,
            "marp_slides_markdown": result.content,
            "provider_used": result.provider_used,
            "duration_sec": result.duration_sec,
            "token_usage": result.token_usage,
        }
    except Exception as exc:
        log.exception("Error generating slide deck")
        raise HTTPException(status_code=500, detail=str(exc))

# ── Phase 4: Thesis Defense Simulator ─────────────────────────────────────────

@app.post("/api/v1/academic/defense/questions", tags=["Phase 4: Academic AI"])
def get_defense_questions(req: DefenseQuestionsRequest):
    """Generates 3 tough viva interrogation questions from the committee."""
    try:
        from agents.defense_simulator_agent import generate_defense_questions
        return generate_defense_questions(paper_title=req.title, paper_content=req.content)
    except Exception as exc:
        log.exception("Error generating defense questions")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/academic/defense/evaluate", tags=["Phase 4: Academic AI"])
def evaluate_defense(req: DefenseEvaluateRequest):
    """Evaluates student rebuttal and returns scores and academic verdict."""
    try:
        from agents.defense_simulator_agent import evaluate_defense_rebuttal
        result = evaluate_defense_rebuttal(
            examiner_name=req.examiner_name,
            examiner_title=req.examiner_title,
            question=req.question,
            student_answer=req.student_answer,
        )
        if result.get("error"):
            raise HTTPException(status_code=503, detail="Evaluation unavailable, please retry")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Error evaluating defense rebuttal")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Phase 4: Temporal Fact-Checker (Live Web Grounding) ──────────────────────

@app.post("/api/v1/academic/fact-check/temporal", tags=["Phase 4: Academic AI"])
def run_temporal_fact_check(req: TemporalFactCheckRequest):
    """Verifies paper claims against live 2024-2026 web benchmarks using You.com."""
    try:
        from agents.fact_check_agent import verify_paper_claims_live
        return verify_paper_claims_live(paper_title=req.title, paper_content=req.content)
    except Exception as exc:
        log.exception("Error running temporal fact-check")
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

