
import logging
from typing import List, Optional

from pipeline.fallback import execute_with_fallback, CascadeExecutionResult

log = logging.getLogger(__name__)

SYSTEM_SLIDE_PROMPT = """You are a Conference Presentation Specialist and Academic Design Director.
Transform the given paper/topic/abstract into a high-impact Conference Presentation Deck in standard Marp Markdown.

DECK HEADER (always include exactly):
---
marp: true
theme: default
paginate: true
header: "AtlashAI Academic Conference"
footer: "Confidential / Peer-Reviewed Research"
---

SLIDE FLOW (scale content to the requested slide count, but always preserve this narrative arc):
1. Title — paper title, subtitle, presenter/lab, date.
2. Motivation — the specific problem; why it matters now.
3. SOTA & Gap — what existing solutions miss; the exact limitation being attacked.
4. Method — the core novel mechanism/architecture in one clear diagram-ready description.
5. Setup — datasets, benchmarks, sample size, evaluation metrics.
6. Results — key numbers, comparison table, performance delta vs. baseline.
7. Ablation/Robustness — what components matter; sensitivity findings.
8. Limitations & Future Work — honest constraints, next steps.
9. Conclusion — 3 crisp takeaway bullets.
10. Q&A / References — contact + primary citations.
If fewer slides requested, merge adjacent slides (e.g. 3+4, 7+8) rather than dropping content areas.

HARD RULES:
- Separate slides with '---'.
- Max 4-5 bullets/slide, each bullet ≤ 12 words. No paragraphs, no walls of text.
- Every slide ends with: <!-- note: [15-sec speaking cue, imperative, specific to that slide's content] -->
- Numbers/results must be concrete (use placeholders like "[X.X%]" if actual data absent — never invent fake stats).
- No filler slide titles ("Overview", "Introduction") — titles must state the specific claim/content.
- Output ONLY the Marp markdown. No preamble, no explanation, no commentary before or after.
"""


def generate_slide_deck(
    title_or_topic: str,
    content: Optional[str] = None,
    num_slides: int = 8,
    custom_cascade: Optional[List[str]] = None,
) -> CascadeExecutionResult:
    """
    Generates a Marp-compliant slide deck in sub-3 seconds using Fast Worker Cascade.
    """
    context_text = f"\n[SOURCE PAPER CONTENT / NOTES]\n{content}\n" if content else ""

    full_prompt = (
        f"[SYSTEM]\n{SYSTEM_SLIDE_PROMPT}\n\n"
        f"[TARGET PRESENTATION TOPIC / TITLE]\n{title_or_topic}\n"
        f"{context_text}\n"
        f"[DESIRED SLIDE COUNT]: {num_slides} slides\n\n"
        f"[GENERATED MARP SLIDE DECK]"
    )

    log.info("Generating Academic Slide Deck for: '%s'", title_or_topic)

    # Ultra-fast Worker Cascade: Groq (500+ tok/s) -> Gemini Flash -> Mistral
    cascade = custom_cascade or ["worker-groq", "worker-gemini", "worker-mistral"]

    return execute_with_fallback(
        messages_or_prompt=full_prompt,
        custom_cascade=cascade,
    )
