"""
fact_check_agent.py

Verifies key claims in the generated report against the original scraped
sources. The agent returns structured verification metadata only; it does not
rewrite the final report. Final report cleanup is deterministic Python logic.
"""

import json
import logging
import re
from typing import Dict

log = logging.getLogger(__name__)

_FALLBACK_FACT_CHECK = {
    "supported": 0,
    "unsupported": 0,
    "partial": 0,
    "confidence": 0.8,
    "unsupported_phrases": [],
}

_FACT_CHECK_PROMPT = """\
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

SOURCE MATERIAL:
{sources}

REPORT TO FACT-CHECK:
{report}
"""


def run_fact_check_node(state: Dict, llm) -> Dict:
    """
    LangGraph node: checks the writer's report against scraped + RAG sources.

    Adds to state:
        fact_check_result - raw structured LLM output
        report            - sanitized writer report
        fact_check_score  - confidence float 0-1 from JSON
    """
    report: str = state.get("report", "").strip()
    if not report:
        log.warning("FactCheck: no report to verify, skipping")
        return {**state, "fact_check_result": "", "fact_check_score": None}

    search_res = state.get("search_results", "")
    if isinstance(search_res, list):
        search_res = "\n\n".join(search_res)

    sources = "\n\n".join(
        filter(
            None,
            [
                state.get("scraped_content", "")[:3000],
                state.get("rag_context", "")[:1500],
                search_res[:1000],
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
            "fact_check_score": 0.5,
        }

    try:
        prompt = _FACT_CHECK_PROMPT.format(
            sources=sources,
            report=cleaned_report[:4000],
        )
        result: str = llm.invoke(prompt).content
        log.info("FactCheck: result length=%d chars", len(result))

        fact_data = _parse_fact_check_json(result)
        score = float(fact_data.get("confidence", 0.8))

        return {
            **state,
            "fact_check_result": result,
            "report": cleaned_report,
            "fact_check_score": score,
        }

    except Exception as exc:
        log.exception("FactCheck LLM call failed")
        return {
            **state,
            "fact_check_result": f"FactCheck error: {exc}",
            "report": cleaned_report,
            "fact_check_score": 0.5,
        }


def _parse_fact_check_json(result: str) -> Dict:
    """Parse the fact-check JSON object, tolerating accidental fenced output."""
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

    if not isinstance(data, dict):
        return dict(_FALLBACK_FACT_CHECK)

    try:
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
    except (TypeError, ValueError):
        data["confidence"] = 0.8

    if not isinstance(data.get("unsupported_phrases"), list):
        data["unsupported_phrases"] = []

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
