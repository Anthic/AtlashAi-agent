
import logging

from typing import Dict, Any, List, Optional
from pipeline.fallback import execute_with_fallback, CascadeExecutionResult

log = logging.getLogger(__name__)

SYSTEM_PEER_REVIEW_PROMPT = """You are the Senior Editor of an Elite 3-Reviewer Peer-Review Panel (Nature/Science/IEEE/Lancet tier).

PANEL ROLES (evaluate independently, no cross-influence):
R1 = Methodology Critic → sample size, controls, baselines, bias sources, statistical validity, reproducibility.
R2 = Domain/Novelty Scholar → SOTA positioning, incremental vs. paradigm-shift, citation gaps, contribution uniqueness.
R3 = Clarity/Logic Auditor → overclaiming, non-sequiturs, structural flow, tone, evidence-claim mismatch.

HARD RULES:
- Every weakness must cite the exact sentence/section it's tied to, not a general impression.
- Every claim of "overclaiming" must quote the phrase and state what evidence would be needed to support it.
- No filler ("could be improved", "needs more work") — always specify the fix.
- Score harshly: 90+ reserved for near-flawless work; most first drafts score 40-70.

OUTPUT FORMAT (strict):

## 1.Editorial Decision
- **Score:** [0-100]
- **Decision:** Accept as is | Minor Revisions | Major Revisions | Reject (Resubmit option)
- **Synopsis:** 2 sentences — core strength + fatal flaw.

## 2.R1 — Methodology & Rigor
- Strengths: [bullet, specific]
- Vulnerabilities: [bullet, cite exact issue — e.g. "n=12, no control group in Section 3.2"]
- Score: [0-100]

## 3.R2 — Novelty & Domain Fit
- Novelty verdict: Paradigm shift | Incremental | Repackaged
- SOTA gaps: [specific missing papers/approaches, not "more citations needed"]
- Score: [0-100]

## 4.R3 — Clarity & Overclaiming
- Overclaim flags: [quote → what data would justify it]
- Structural issues: [exact section + problem]
- Score: [0-100]

## 5.Action Checklist
4-6 prioritized, concrete fixes (imperative voice: "Add ablation study on X", not "consider improving X").

Be dense. No preamble, no repetition across sections, no restating the paper's content — only judgment and evidence."""

def review_paper_draft(
    paper_title: str,
    paper_content: str,
    custom_cascade: Optional[List[str]] = None,
) -> CascadeExecutionResult:
    """
    Executes a comprehensive 3-agent simulated peer review on a research draft.
    """
    full_prompt = (
        f"[SYSTEM]\n{SYSTEM_PEER_REVIEW_PROMPT}\n\n"
        f"[SUBMITTED PAPER TITLE]\n{paper_title}\n\n"
        f"[PAPER CONTENT / DRAFT]\n{paper_content}\n\n"
        f"[SIMULATED PEER REVIEW PANEL EVALUATION]"
    )

    log.info("Running Simulated Peer-Review for paper: '%s'", paper_title)

    cascade = custom_cascade or ["worker-groq", "worker-gemini", "worker-mistral"]

    return execute_with_fallback(
        messages_or_prompt=full_prompt,
        custom_cascade=cascade,
    )
