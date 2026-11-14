"""
summarizer_agent.py
────────────────────────────────────────────────────────────
Condenses long scraped content into focused bullet-point key facts
before it enters the writer chain.

Why it matters
──────────────
Raw scraped pages can be thousands of tokens of boilerplate (nav bars,
ads, repeated headers). Summarising first:
  • Reduces writer prompt size → cheaper & faster
  • Focuses the LLM on signal, not noise
  • Improves citation accuracy
"""

import logging
import re
from typing import Dict

from pipeline.fallback import invoke_with_cascade
from tools.safety import wrap_untrusted, UNTRUSTED_NOTICE

log = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are a research extraction specialist.

Given raw web content scraped from multiple sources, extract and organise \
the most relevant facts for the topic provided.

Rules:
- Focus ONLY on information relevant to the topic
- Extract specific data points: numbers, dates, names, statistics
- Keep the [Source N] number shown before each source on every fact
- Be concise — bullet points only (max 15 bullets total)
- Discard boilerplate, navigation text, and ads

Topic: {topic}

{untrusted_notice}

Raw Content:
{raw_content}

Output format:
### Key Facts
- [Source N] <fact>        # N = the number shown before each source, e.g. [Source 3]
- ...
"""

SUMMARY_INPUT_CHARS = 9000


def _balanced_excerpt(raw: str, total_chars: int = SUMMARY_INPUT_CHARS) -> str:
    """Give every scraped source an equal share instead of cutting off the tail."""
    blocks = [b for b in re.split(r"(?=^\[Source \d+\] )", raw, flags=re.M) if b.strip()]
    if len(blocks) <= 1:
        return raw[:total_chars]
    per = max(600, total_chars // len(blocks))
    return "\n\n".join(b[:per] for b in blocks)


def run_summarizer_node(state: Dict, llm) -> Dict:
    """
    LangGraph node: summarises scraped_content into condensed bullet points.

    Adds / updates state:
        summarized_content — condensed key facts (replaces scraped in writer)
    """
    raw: str = state.get("scraped_content", "").strip()
    topic: str = state.get("topic", "").strip()

    if not raw:
        log.warning("Summarizer: no scraped content, skipping")
        return {**state, "summarized_content": ""}

    # Skip summarisation if content is already short enough
    if len(raw) < 1200:
        log.info("Summarizer: content already short (%d chars), skipping", len(raw))
        return {**state, "summarized_content": raw}

    try:
        prompt = _SUMMARIZE_PROMPT.format(
            topic=topic,
            untrusted_notice=UNTRUSTED_NOTICE,
            raw_content=wrap_untrusted(_balanced_excerpt(raw), "scraped_web_pages"),
        )
        summary: str = invoke_with_cascade(llm, prompt, tier="worker").strip()
        log.info("Summarizer: %d chars → %d chars", len(raw), len(summary))
        return {**state, "summarized_content": summary}

    except Exception as exc:
        log.exception("Summarizer failed")
        return {**state, "summarized_content": raw, "error": f"Summarizer failed: {exc}"}
