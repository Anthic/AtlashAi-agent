"""
fact_check_agent.py

Verifies key claims in the generated report against the original scraped
sources. The agent returns structured verification metadata only; it does not
rewrite the final report. Final report cleanup is deterministic Python logic.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from tools.you_search_tool import search_temporal_facts
from tools.safety import wrap_untrusted, build_messages
from pipeline.fallback import invoke_with_cascade
log = logging.getLogger(__name__)

SOURCE_CHARS = 9000     # scraped text shown to the checker (was 3000)
REPORT_CHARS = 12000    # report text checked (was 4000)
MAX_SENTENCES_REMOVED = 3

# Returned when the model output cannot be parsed: NOT a verification result.
_FALLBACK_FACT_CHECK = {
    "supported": 0,
    "unsupported": 0,
    "partial": 0,
    "confidence": None,
    "parsed": False,
    "unsupported_phrases": [],
}

_FACT_CHECK_PROMPT = """
You are a rigorous fact-checking assistant.

Your job is to verify every factual claim in the REPORT against the SOURCE
MATERIAL provided. Do NOT rely on your training data; only check against the
sources given.

Instructions:
1. Extract up to 10 key factual claims from the report.
2. For each claim, search the source material for supporting evidence.
3. Mark each claim as:
   - SUPPORTED: evidence found in sources
   - UNSUPPORTED: no evidence found in the provided sources
   - PARTIAL: partially supported and needs qualification
4. Do not rewrite the report.
5. Do not output markdown, prose, claim tables, corrected sections, or
   placeholders.
6. Return ONLY this JSON object:

{{
  "supported": <int>,
  "unsupported": <int>,
  "partial": <int>,
  "confidence": <0-1 float>,
  "unsupported_phrases": ["exact unsupported sentence or heading from the report"]
}}

SOURCE MATERIAL (untrusted data):
{sources}

REPORT TO FACT-CHECK:
{report}
"""


def _score_from(data: Dict) -> Optional[float]:
    """Real score derived from the per-claim counts; None when nothing was verified."""
    if not data.get("parsed"):
        return None
    sup, uns, par = (int(data.get(k, 0) or 0) for k in ("supported", "unsupported", "partial"))
    total = sup + uns + par
    if total > 0:
        return round((sup + 0.5 * par) / total, 3)
    conf = data.get("confidence")
    return conf if isinstance(conf, (int, float)) else None


def _strip_unsupported(report: str, phrases: List[str]) -> str:
    """
    Remove sentences the checker flagged as unsupported (exact, normalised match only;
    never headings; capped so a bad checker response cannot hollow out the report).
    """
    removed = 0
    wanted = [re.sub(r"\s+", " ", p).strip().lower() for p in phrases if isinstance(p, str) and len(p.strip()) >= 30]
    if not wanted:
        return report
    out_lines = []
    for line in report.splitlines():
        if removed >= MAX_SENTENCES_REMOVED or re.match(r"^\s*#{1,6}\s", line) or not line.strip():
            out_lines.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = []
        for sent in parts:
            norm = re.sub(r"\s+", " ", re.sub(r"\[Source \d+\]", "", sent)).strip().lower()
            if removed < MAX_SENTENCES_REMOVED and norm and any(w == norm or w in norm for w in wanted):
                removed += 1
                continue
            kept.append(sent)
        if kept:
            out_lines.append(" ".join(kept))
    if removed:
        log.info("FactCheck: removed %d unsupported sentence(s)", removed)
    return "\n".join(out_lines)


def run_fact_check_node(state: Dict, llm) -> Dict:
    """
    LangGraph node: checks the writer's report against scraped + RAG sources.

    Adds to state:
        fact_check_result - raw structured LLM output
        report            - sanitised report (unsupported sentences removed, capped)
        fact_check_score  - 0-1 score computed from per-claim counts, or None when the
                            checker output could not be parsed ("not verified")
        fact_check_parsed - True only if the score is a real verification result
    """
    report: str = state.get("report", "").strip()
    if not report:
        log.warning("FactCheck: no report to verify, skipping")
        return {**state, "fact_check_result": "", "fact_check_score": None, "fact_check_parsed": False}

    search_res = state.get("search_results", "")
    if isinstance(search_res, list):
        search_res = "\n\n".join(dict.fromkeys(search_res))

    sources = "\n\n".join(
        filter(
            None,
            [
                state.get("scraped_content", "")[:SOURCE_CHARS],
                state.get("rag_context", "")[:1500],
                search_res[:1500],
            ],
        )
    )

    cleaned_report = _sanitize_final_report(report)

    if not sources.strip():
        log.warning("FactCheck: no sources available, skipping")
        return {
            **state,
            "fact_check_result": "No sources to verify against.",
            "report": cleaned_report,
            "fact_check_score": None,
            "fact_check_parsed": False,
        }

    try:
        prompt = _FACT_CHECK_PROMPT.format(
            sources=wrap_untrusted(sources, "source_material"),
            report=cleaned_report[:REPORT_CHARS],
        )
        result: str = invoke_with_cascade(llm, build_messages(
            "You are a rigorous fact-checking assistant. Output only the requested JSON.", prompt))
        log.info("FactCheck: result length=%d chars", len(result))

        fact_data = _parse_fact_check_json(result)
        score = _score_from(fact_data)
        final_report = cleaned_report
        if fact_data.get("parsed"):
            final_report = _strip_unsupported(cleaned_report, fact_data.get("unsupported_phrases", []))

        return {
            **state,
            "fact_check_result": result,
            "report": final_report,
            "fact_check_score": score,
            "fact_check_parsed": bool(fact_data.get("parsed")),
        }

    except Exception as exc:
        log.exception("FactCheck LLM call failed")
        return {
            **state,
            "fact_check_result": f"FactCheck error: {exc}",
            "report": cleaned_report,
            "fact_check_score": None,
            "fact_check_parsed": False,
        }


def _parse_fact_check_json(result: str) -> Dict:
    """Parse the fact-check JSON object, tolerating accidental fenced output.
    On failure returns parsed=False (never a made-up confidence)."""
    raw = result.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        raw = match.group(0) if match else raw

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return dict(_FALLBACK_FACT_CHECK)

    if not isinstance(data, dict) or not any(k in data for k in ("supported", "unsupported", "partial", "confidence")):
        return dict(_FALLBACK_FACT_CHECK)

    try:
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"]))) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        data["confidence"] = None

    if not isinstance(data.get("unsupported_phrases"), list):
        data["unsupported_phrases"] = []
    data["parsed"] = True
    return data


def _sanitize_final_report(report: str) -> str:
    """Remove internal LLM artifacts without asking the LLM to rewrite the report."""
    if not report:
        return ""

    cleaned = re.split(r"\n\s*(?:JSON Summary|JSON\s*:)\s*\n", report, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"```json\s*\{.*?\}\s*```", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*---+\s*$", "", cleaned, flags=re.MULTILINE)

    output = []
    artifact_patterns = (
        "removed due to lack of source evidence",
        "not found in sources",
        "[unverified]",
    )

    for line in cleaned.splitlines():
        lowered = line.lower()
        if any(pattern in lowered for pattern in artifact_patterns):
            if output and re.match(r"^\s*#{2,6}\s+", output[-1]):
                output.pop()
            continue
        output.append(line)

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


TEMPORAL_VERIFY_PROMPT = """You are a World-Class Academic Temporal Fact-Checker.
Your job is to examine whether the following specific CLAIM from an academic paper is still considered TRUE, SOTA, or VALID today (2024-2026), or if it has been SURPASSED, CONTRADICTED, or DISPUTED.
PAPER TOPIC: {topic}
CLAIM: {claim}
LIVE WEB EVIDENCE (from You.com search):
{evidence}
Analyze the claim against the live web evidence.
Return strictly a JSON object:
{{
  "claim": "{claim}",
  "status": "VERIFIED_CURRENT" | "SURPASSED_OR_OUTDATED" | "DISPUTED" | "UNVERIFIED",
  "confidence": 0.85,
  "explanation": "Clear 2-sentence explanation of why it is current or outdated.",
  "recent_sota_alternative": "Name of any newer 2024-2026 model/method that surpassed it, or 'None'"
}}
"""
def verify_paper_claims_live(paper_title: str, paper_content: str, llm=None) -> Dict[str, Any]:
    """
    Extracts high-impact claims from paper and runs live temporal verification using You.com.
    """
    from pipeline.fallback import execute_with_fallback
   
    extract_prompt = f"""From this paper titled "{paper_title}", extract up to 3 boldest empirical or SOTA claims (e.g., 'achieves highest accuracy', 'first framework to solve X', 'surpasses model Y').
Return strictly a JSON list of strings, e.g. ["Claim 1", "Claim 2"].
Paper excerpt:
{paper_content[:4000]}
"""
    res = execute_with_fallback(build_messages(
        "You extract claims from academic papers. Output only the requested JSON list.",
        extract_prompt.replace(paper_content[:4000], wrap_untrusted(paper_content[:4000], "paper")),
    ))
    claims = []
    try:
        match = re.search(r"\[.*?\]", res.content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            claims = [c for c in parsed if isinstance(c, str) and c.strip()]
    except Exception:
        log.warning("Temporal fact-check: could not parse claim list")
    if not claims:
        # Do not invent a claim (the old fallback fabricated one and then "verified" it).
        return {
            "title": paper_title,
            "verified_claims": [],
            "temporal_sources": [],
            "error": "no_claims_extracted",
            "timestamp": time.strftime("%Y-%m-%d"),
        }
    results = []
    all_citations = []
    
    for claim in claims[:3]:
        hits = search_temporal_facts(claim, topic=paper_title)
        evidence_text = ""
        citations = []
        for h in hits:
            citations.append({"title": h["title"], "url": h["url"]})
            all_citations.append(h["url"])
            evidence_text += f"- {h['title']}: {' '.join(h.get('snippets', [])[:2])}\n"
        check_prompt = TEMPORAL_VERIFY_PROMPT.format(
            topic=paper_title,
            claim=claim,
            evidence=wrap_untrusted(evidence_text, "web_evidence") if evidence_text else "No external evidence found."
        )
        eval_res = execute_with_fallback(build_messages(
            "You are an academic temporal fact-checker. Output only the requested JSON.", check_prompt))
        match = re.search(r"\{.*\}", eval_res.content, re.DOTALL)
        if match:
            try:
                item = json.loads(match.group(0))
                item["citations"] = citations
                results.append(item)
                continue
            except Exception:
                pass
        # Parse failure => UNVERIFIED (never a fabricated "VERIFIED_CURRENT 0.75")
        results.append({
            "claim": claim,
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "explanation": "The verification result could not be parsed; this claim was not checked.",
            "recent_sota_alternative": "None",
            "citations": citations
        })
    return {
        "title": paper_title,
        "verified_claims": results,
        "temporal_sources": list(set(all_citations)),
        "timestamp": time.strftime("%Y-%m-%d")
    }