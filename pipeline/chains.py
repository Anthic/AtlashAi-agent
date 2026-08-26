"""
pipeline/chains.py
────────────────────────────────────────────────────────────
Writer, Critic and Anti-Cutoff chains powered by Resilient Fallback Cascade.
Zero hallucination on continuation + Zero premature sentence truncation.
"""

import logging
import re
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from pipeline.fallback import execute_with_fallback

log = logging.getLogger(__name__)

_RE_JSON_SPLIT = re.compile(r"\n\s*(?:JSON Summary|JSON\s*:)\s*\n", re.IGNORECASE)
_RE_JSON_BLOCK = re.compile(r"```json\s*\{.*?\}\s*```", re.IGNORECASE | re.DOTALL)
_RE_HLINE      = re.compile(r"^\s*---+\s*$", re.MULTILINE)
_RE_HEADING    = re.compile(r"^\s*#{2,6}\s+")
_RE_EXCESS_NL  = re.compile(r"\n{3,}")

# Detects a line that is clearly a table row, a source/bullet URL line, or a
# closed code fence — these are legitimate ways for a report to end and
# should never be flagged as a mid-sentence cutoff.
_RE_COMPLETE_LINE = re.compile(
    r"^\s*(\|.*\|\s*$|```\s*$|[-*]\s*https?://\S+\s*$|https?://\S+\s*$)"
)

# ── Writer Prompt ─────────────────────────────────────────────────────────────

_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert research writer. Your output will be \
fact-checked, so precision is critical.

STRICT RULES:
1. ONLY use facts from the provided sources below. Never invent data.
2. If a fact is not in the provided context, omit that claim or subsection entirely.
3. ONLY cite URLs from the verified_urls list — never modify or create URLs.
4. Every claim must have a [Source N] citation.
5. If sources conflict, mention both and prefer the most recent one.
6. Use clear, professional language. Avoid filler phrases.
7. Do NOT include any horizontal rules (---) between sections.
8. Never write placeholders such as "Not found in sources" or "[UNVERIFIED]".
9. Do NOT append any conversational preamble, postamble, or JSON summary. Your report must end strictly with the Conclusion or Sources section.""",
        ),
        (
            "human",
            """Topic:
{topic}

Key Facts (Summarised):
{summarized_content}

Raw Search Results:
{search_results}

RAG Context:
{rag_context}

{critique_section}

Only use these verified URLs in citations and in the Sources section:
{verified_urls}

Output format:
## Introduction
## Key Findings
### Finding 1
### Finding 2
### Finding 3
## Conclusion
## Sources""",
        ),
    ]
)

# ── Critic Prompt ─────────────────────────────────────────────────────────────

_critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict but constructive critic for research reports. "
            "Be precise — your score drives the rewrite loop.",
        ),
        (
            "human",
            """Review the report below and score it.

Report:
{report}

Return exactly:
Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
...
""",
        ),
    ]
)


# ── Sanitizer ─────────────────────────────────────────────────────────────────

def sanitize_final_report(report: str) -> str:
    if not report:
        return ""
    cleaned = _RE_JSON_SPLIT.split(report)[0]
    cleaned = _RE_JSON_BLOCK.sub("", cleaned)
    cleaned = _RE_HLINE.sub("", cleaned)
    output = []
    artifact_patterns = (
        "removed due to lack of source evidence",
        "not found in sources",
        "[unverified]",
    )
    for line in cleaned.splitlines():
        lowered = line.lower()
        if any(pattern in lowered for pattern in artifact_patterns):
            if output and _RE_HEADING.match(output[-1]):
                output.pop()
            continue
        output.append(line)
    cleaned = "\n".join(output)
    cleaned = _RE_EXCESS_NL.sub("\n\n", cleaned).strip()
    return cleaned


def _looks_complete(text: str) -> bool:
    """
    True if `text` looks like it ends at a legitimate stopping point:
    normal sentence punctuation, a closed code fence, a markdown table row,
    or a bullet/plain source URL line.

    Trailing quote/bracket characters (e.g. a closing " after a period) are
    stripped before the punctuation check, since a model may legitimately
    end a quoted clause with `."` and that's still a complete sentence.

    Deliberately does NOT treat bare ')', ']', '>' or '|' as complete on
    their own — those are far too common mid-sentence (e.g. "...as shown
    (Smith, 2023" or "...value is >10") and would cause false "complete"
    detections.
    """
    if not text:
        return True
    stripped = text.rstrip()
    # Peel off trailing closing-quote/bracket characters before checking
    # for sentence-ending punctuation, e.g. `disorders."` should count as
    # ending in `.` even though the literal last character is `"`.
    core = stripped.rstrip("\"'”’)]")
    if core.endswith((".", "!", "?", "।")):
        return True
    lines = stripped.splitlines()
    last_line = lines[-1] if lines else ""
    return bool(_RE_COMPLETE_LINE.match(last_line))


def _strip_overlap(preceding_text: str, new_chunk: str, max_check_chars: int = 150) -> str:
    """
    If `new_chunk` begins by re-stating a chunk of text that's already at
    the end of `preceding_text` (a model echoing the anchor it was shown),
    trim that repeated prefix off so it isn't duplicated in the joined
    output. Only checks a short trailing/leading window for performance
    and to avoid accidentally trimming a legitimate short repeated word.
    """
    tail = preceding_text[-max_check_chars:].lower()
    head = new_chunk.lower()

    # Try progressively shorter overlaps (longest match first) so a full
    # repeated sentence is caught before falling back to smaller ones.
    max_overlap = min(len(tail), len(head))
    for overlap_len in range(max_overlap, 19, -1):  # ignore trivial overlaps < 20 chars
        if tail[-overlap_len:] == head[:overlap_len]:
            return new_chunk[overlap_len:].lstrip()
    return new_chunk


# ── Zero Cut-off & Grounded Continuation Engine ───────────────────────────────

def run_with_zero_cutoff(
    initial_text: str,
    context: str,
    verified_urls_text: str,
    topic: str = "",
    tier: str = "master",
    max_continuations: int = 2,
) -> str:
    """
    Ensures complete generation without premature sentence cutoff.
    Strictly grounded with context and verified URLs to avoid hallucination.

    Continuation calls use the SAME `tier` as the original generation, so a
    master-tier report doesn't silently drop to worker-tier quality partway
    through.
    """
    text = initial_text.strip()
    truncated = False
    iterations = 0

    while iterations < max_continuations:
        if _looks_complete(text):
            break

        truncated = True
        log.warning("Detected incomplete text ending. Triggering grounded anti-cutoff continuation...")
        anchor = text[-120:]
        # NOTE: the anchor is shown inside a fenced code block, NOT wrapped
        # in quotation marks / ellipsis. Wrapping it in "..." previously
        # caused the model to echo those literal quote/ellipsis characters
        # (and often the whole anchor sentence) back as if it were content
        # to reproduce, which duplicated text and also broke the
        # completion check (trailing `"` instead of `.`).
        continuation_prompt = f"""You are continuing a research report that was cut off, possibly mid-word or mid-sentence, due to token limits.

Topic:
{topic}

Context & Sources:
{context[:2000]}

Verified URLs (ONLY cite from this list — never invent or modify a URL):
{verified_urls_text}

The text below is the END of what has already been written. It is shown ONLY
for you to see where to pick up — do NOT reproduce, quote, or repeat any part
of it in your answer:
---BEGIN ALREADY-WRITTEN TEXT (do not repeat)---
{anchor}
---END ALREADY-WRITTEN TEXT---

STRICT INSTRUCTIONS:
1. Only use facts already present in the Context & Sources above — never invent new data.
2. If the cutoff is mid-word, first complete that exact word, then continue.
3. Your answer must contain ONLY new text — do not repeat, quote, or paraphrase anything from the already-written text shown above.
4. Do not wrap your answer in quotation marks.
5. Complete the thought and conclude the section cleanly.
6. Do NOT add any preamble, apology, or meta-commentary — output only the continuation text."""

        try:
            cont_res = execute_with_fallback(continuation_prompt, tier=tier)
            cont_chunk = cont_res.content.strip()
            # Defensively strip a leading/trailing wrapping quote in case
            # the model still wraps its answer despite instruction #4.
            if len(cont_chunk) >= 2 and cont_chunk[0] in "\"'" and cont_chunk[-1] in "\"'":
                cont_chunk = cont_chunk[1:-1].strip()
            if not cont_chunk:
                break

            # Guard against the model still echoing a chunk of the anchor
            # verbatim at the start of its continuation.
            cont_chunk = _strip_overlap(text, cont_chunk)
            if not cont_chunk:
                break

            # Word-boundary-aware join: don't insert a space if the
            # continuation is punctuation, or if either side already has
            # whitespace at the boundary.
            if text and not text[-1].isspace() and not cont_chunk[0].isspace() \
                    and not cont_chunk.startswith((".", ",", "!", "?", ")", "]")):
                text += " " + cont_chunk
            else:
                text += cont_chunk
            truncated = False  # optimistically clear; re-checked at loop top

        except Exception as exc:
            log.warning("Continuation failed: %s", exc)
            break

        iterations += 1

    if truncated and not _looks_complete(text):
        log.warning(
            "Report may still be incomplete after %d continuation attempt(s).",
            iterations,
        )

    return sanitize_final_report(text)


# ── Fallback-Powered Node Runners ─────────────────────────────────────────────

def run_writer(state: dict, chain=None) -> dict:
    """Fallback-powered Writer Node."""
    if state.get("error") and not state.get("scraped_content"):
        return {**state, "report": "", "critique": ""}

    try:
        topic = state.get("topic", "")
        verified_urls = state.get("verified_urls", [])
        verified_url_text = "\n".join(f"- {url}" for url in verified_urls) or "- None"

        summarized = state.get("summarized_content", "").strip()
        raw_results = state.get("search_results", [])
        if isinstance(raw_results, str):
            raw_results = [raw_results] if raw_results else []
        search_results = "\n\n".join(raw_results)[:2000]
        rag_context = state.get("rag_context", "")[:1500]

        critique_feedback = state.get("critique", "")
        critique_section = (
            f"Previous critique — improve based on this feedback:\n{critique_feedback}"
            if critique_feedback else ""
        )

        # 1. Format the prompt
        formatted_messages = _writer_prompt.format_messages(
            topic=topic,
            summarized_content=summarized or state.get("scraped_content", "")[:4000],
            search_results=search_results,
            rag_context=rag_context,
            critique_section=critique_section,
            verified_urls=verified_url_text,
        )

        # 2. Execute via Resilient Master Cascade
        res = execute_with_fallback(formatted_messages, tier="master")
        raw_report = res.content

        # 3. Pass through Zero Cut-off & Grounded Sanitizer.
        # Continuation runs at the SAME tier as the original write, and
        # gets the topic + search results + rag context so it doesn't lose
        # the thread if summarized_content happens to be empty.
        full_context = "\n\n".join(
            part for part in (summarized, search_results, rag_context) if part
        )
        final_report = run_with_zero_cutoff(
            initial_text=raw_report,
            context=full_context,
            verified_urls_text=verified_url_text,
            topic=topic,
            tier="master",
        )

        log.info("Writer: report generated via [%s] (%d chars)", res.provider_used, len(final_report))
        return {**state, "report": final_report}

    except Exception as exc:
        log.exception("Writer failed across all fallbacks")
        return {**state, "report": "", "error": f"Writer failed: {exc}"}


def run_critic(state: dict, chain=None) -> dict:
    """Fallback-powered Critic Node."""
    report = state.get("report", "")
    if not report:
        return {**state, "critique": "No report to critique.", "critique_score": 0}

    try:
        # No truncation: cutting the report to a fixed prefix would hide the
        # Conclusion/Sources sections from the critic on longer reports,
        # producing false "missing conclusion"-style critiques and
        # needless rewrite loops on reports that are actually complete.
        formatted_messages = _critic_prompt.format_messages(report=report)
        res = execute_with_fallback(formatted_messages, tier="worker")
        critique = res.content
        log.info("Critic: critique generated via [%s] (%d chars)", res.provider_used, len(critique))
        return {**state, "critique": critique}

    except Exception as exc:
        log.exception("Critic failed across all fallbacks")
        return {**state, "critique": "", "critique_score": 0, "error": f"Critic failed: {exc}"}