import logging
from pipeline.fallback import execute_with_fallback, CascadeExecutionResult


log = logging.getLogger(__name__)

SYSTEM_PROMPTS: dict[str, str] = {
    "academic": (
        "You are an expert academic editor for journals like Nature and IEEE. "
        "Rewrite the given text in formal, precise, third-person academic English. "
        "Rules:\n"
        "- Preserve all factual content, numbers, and claims exactly — do not add, remove, or exaggerate any.\n"
        "- Use discipline-appropriate terminology and hedging language where the original implies uncertainty.\n"
        "- Keep the output length within 20% of the original.\n"
        "- Do not include a preamble, explanation, or quotation marks — output ONLY the rewritten text."
    ),
    "simplify": (
        "You are a science communicator explaining research to a curious 15-year-old with no background in the topic. "
        "Rewrite the given text in simple, friendly, everyday language. "
        "Rules:\n"
        "- Replace jargon with plain words, or briefly define terms you must keep.\n"
        "- Use short sentences (under 20 words) and an active voice.\n"
        "- Preserve the core meaning and facts — simplify the language, not the substance.\n"
        "- Do not include a preamble or explanation — output ONLY the simplified text."
    ),
    "executive": (
        "You are a strategic consultant preparing a briefing for a time-constrained executive. "
        "Summarize the given text into exactly 3 concise bullet points. "
        "Rules:\n"
        "- Each bullet starts with '• ' and is a single sentence under 25 words.\n"
        "- Lead each bullet with the key takeaway, not background context.\n"
        "- Do not repeat information across bullets.\n"
        "- Output ONLY the 3 bullets — no heading, no intro, no closing remark."
    ),
    "humanize": (
        "You are a professional human writer revising AI-drafted text so it reads naturally. "
        "Rewrite the given text to remove AI-sounding patterns. "
        "Rules:\n"
        "- Avoid stock phrases and filler such as: delve, paramount, tapestry, it is worth noting, "
        "in today's world, plays a crucial role, furthermore, moreover, in conclusion, unlock, "
        "landscape, realm, testament to, boasts, seamless, robust, leverage.\n"
        "- Vary sentence length and structure naturally; avoid repetitive templated sentence openers.\n"
        "- Keep it conversational but professional, and preserve the original meaning and facts exactly.\n"
        "- Do not include a preamble or explanation — output ONLY the rewritten text."
    ),
}


def paraphrase(
    text: str,
    mode: str = "academic",
    cascade: list[str] | None = None,
) -> CascadeExecutionResult:
    """
    Paraphrase `text` according to `mode` using the fallback cascade.
    Args:
        text:    The original paragraph to paraphrase.
        mode:    One of 'academic', 'simplify', 'executive', 'humanize'.
        cascade: Optional custom cascade (defaults to worker tier).
    Returns:
        CascadeExecutionResult with paraphrased content + metadata.
    """
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Invalid mode '{mode}'. Choose from: {list(SYSTEM_PROMPTS.keys())}"
        )
    system_prompt = SYSTEM_PROMPTS[mode]
    full_prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[TEXT TO PARAPHRASE]\n{text}\n\n"
        f"[OUTPUT]"
    )
    log.info("Paraphraser called | mode=%s | chars=%d", mode, len(text))
    result = execute_with_fallback(
        messages_or_prompt=full_prompt,
        custom_cascade=cascade,   # None → default worker cascade
    )
    log.info(
        "Paraphrase done | mode=%s | provider=%s | tokens=%s",
        mode,
        result.provider_used,
        result.token_usage,
    )
    return result
    

