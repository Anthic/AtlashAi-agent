"""
agents/supervisor_agent.py
────────────────────────────────────────────────────────────
Mistral Large as the Senior Supervisor (Quality Gate only).

Architecture:
  Worker (Groq Qwen / Gemini Flash)  →  writes the full draft fast
  Supervisor (Mistral Large)         →  reviews like a senior engineer:
                                          - Checks factual grounding
                                          - Flags hallucinated claims
                                          - Points out wrong citations
                                          - Identifies off-topic sections
  Worker (same small model)          →  fixes ONLY the flagged issues

This means Mistral Large is called exactly ONCE (not for writing,
only for reviewing). The worker does all the actual generation work.

Used by: pipeline/chains.py → run_writer_node()
"""

import logging
import re
from typing import Optional

from pipeline.fallback import execute_with_fallback

log = logging.getLogger(__name__)

# ── Supervisor Prompt ─────────────────────────────────────────────────────────

_SUPERVISOR_REVIEW_PROMPT = """\
You are a senior research editor. A junior writer has produced a draft report.
Your job is NOT to rewrite it — only to identify specific issues.

Topic: {topic}

Verified Sources Available (ONLY these URLs are legitimate):
{verified_urls}

Draft Report:
{report}

Review the draft and respond in this EXACT format:

VERDICT: APPROVED
(if the report is factually grounded, all citations come from verified URLs, 
and the topic is fully covered — score ≥ 7/10)

OR:

VERDICT: NEEDS_FIX
SCORE: X/10
ISSUES:
- [SECTION: Introduction] Claim "X" has no source citation. Add a [Source N] reference.
- [SECTION: Key Findings] URL "example.com/fake" is not in the verified sources list. Replace with a real URL.
- [SECTION: Conclusion] The paragraph drifts off-topic about Y, which is unrelated to "{topic}". Remove it.

FIX_INSTRUCTIONS:
<Concise, actionable instructions telling the worker exactly what to fix.
Reference specific sections and sentences. Do NOT rewrite anything yourself.>

Rules:
- Be precise. Reference exact sections and sentences.
- Only flag real problems (hallucinated facts, wrong URLs, major topic drift).
- Do NOT nitpick style or word choice.
- If score ≥ 7, always APPROVE even with minor issues.
"""

# ── Fix Prompt (sent back to the small worker model) ──────────────────────────

_WORKER_FIX_PROMPT = """\
You are a research writer. Your supervisor reviewed your draft and found specific issues.
Fix ONLY the identified issues. Do NOT rewrite sections that were not flagged.

Topic: {topic}

Verified URLs (only cite from this list):
{verified_urls}

Your original draft:
{original_report}

Supervisor's fix instructions:
{fix_instructions}

Output the complete corrected report. Keep all approved sections exactly as they are.
Only modify the specific parts mentioned in the fix instructions.
"""


# ── Public API ─────────────────────────────────────────────────────────────────

class SupervisorResult:
    def __init__(self, final_report: str, approved_on_first_try: bool,
                 supervisor_score: int, fix_applied: bool):
        self.final_report = final_report
        self.approved_on_first_try = approved_on_first_try
        self.supervisor_score = supervisor_score
        self.fix_applied = fix_applied


def _parse_supervisor_response(response: str) -> dict:
    """Parse the structured supervisor response."""
    response = response.strip()

    if "VERDICT: APPROVED" in response:
        return {"approved": True, "score": 8, "fix_instructions": ""}

    # Extract score
    score_match = re.search(r"SCORE:\s*(\d+)/10", response)
    score = int(score_match.group(1)) if score_match else 5

    # If score >= 7, treat as approved even if NEEDS_FIX was returned
    if score >= 7:
        return {"approved": True, "score": score, "fix_instructions": ""}

    # Extract fix instructions
    fix_match = re.search(
        r"FIX_INSTRUCTIONS:\s*(.*?)(?:$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    fix_instructions = fix_match.group(1).strip() if fix_match else response

    return {
        "approved": False,
        "score": score,
        "fix_instructions": fix_instructions,
    }


def run_supervisor_review(
    topic: str,
    report: str,
    verified_urls: list,
    worker_tier: str = "worker",
) -> SupervisorResult:
    """
    Mistral Large reviews the draft. If approved → return immediately.
    If rejected (score < 7) → send fix instructions to the worker model.
    Worker fixes ONLY the flagged issues. Mistral is NOT called again.

    Args:
        topic:          Research topic.
        report:         Draft report from the worker writer.
        verified_urls:  List of verified source URLs.
        worker_tier:    The tier to use for the fix pass ("worker" = Groq/Gemini).

    Returns:
        SupervisorResult with the final (possibly corrected) report.
    """
    if not report:
        return SupervisorResult(
            final_report=report,
            approved_on_first_try=False,
            supervisor_score=0,
            fix_applied=False,
        )

    url_text = "\n".join(f"- {u}" for u in verified_urls) if verified_urls else "- None provided"

    # ── Step 1: Mistral Large reviews (ONCE) ─────────────────────────────────
    review_prompt = _SUPERVISOR_REVIEW_PROMPT.format(
        topic=topic,
        verified_urls=url_text,
        report=report[:6000],  # cap to avoid token overflow on very long reports
    )

    try:
        log.info("Supervisor: Mistral Large reviewing draft (%d chars)...", len(report))
        review_res = execute_with_fallback(
            review_prompt,
            tier="master",            # Mistral Large
            call_timeout_sec=25,
        )
        parsed = _parse_supervisor_response(review_res.content)
        log.info(
            "Supervisor verdict: %s | score: %d",
            "APPROVED" if parsed["approved"] else "NEEDS_FIX",
            parsed["score"],
        )
    except Exception as exc:
        log.warning("Supervisor review failed: %s — returning original draft", exc)
        return SupervisorResult(
            final_report=report,
            approved_on_first_try=True,   # assume ok, don't block
            supervisor_score=7,
            fix_applied=False,
        )

    # ── Step 2a: Approved → return immediately ────────────────────────────────
    if parsed["approved"]:
        return SupervisorResult(
            final_report=report,
            approved_on_first_try=True,
            supervisor_score=parsed["score"],
            fix_applied=False,
        )

    # ── Step 2b: Worker fixes ONLY the flagged issues ─────────────────────────
    log.info(
        "Supervisor: sending fix instructions to worker (score=%d). "
        "Worker will fix specific issues only.",
        parsed["score"],
    )

    fix_prompt = _WORKER_FIX_PROMPT.format(
        topic=topic,
        verified_urls=url_text,
        original_report=report[:5000],
        fix_instructions=parsed["fix_instructions"],
    )

    try:
        fix_res = execute_with_fallback(
            fix_prompt,
            tier=worker_tier,         # Groq/Gemini — fast worker fixes it
            call_timeout_sec=20,
        )
        fixed_report = fix_res.content.strip()
        log.info(
            "Worker fix complete via [%s] (%d chars → %d chars)",
            fix_res.provider_used,
            len(report),
            len(fixed_report),
        )
        return SupervisorResult(
            final_report=fixed_report if fixed_report else report,
            approved_on_first_try=False,
            supervisor_score=parsed["score"],
            fix_applied=True,
        )
    except Exception as exc:
        log.warning("Worker fix failed: %s — returning original draft", exc)
        return SupervisorResult(
            final_report=report,
            approved_on_first_try=False,
            supervisor_score=parsed["score"],
            fix_applied=False,
        )
