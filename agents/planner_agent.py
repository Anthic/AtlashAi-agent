import json
import logging
import re
from typing import List, Dict
from pydantic import BaseModel, Field

from pipeline.fallback import execute_with_fallback

log = logging.getLogger(__name__)

# ── Structured output schema ──────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    plan: str = Field(description="3-sentence research plan covering overview, recent advances, and critical analysis")
    sub_questions: List[str] = Field(description="Exactly 4 focused sub-questions that together fully cover the topic")
    search_queries: List[str] = Field(description="One concrete Google-quality search query per sub-question, with year/domain terms")

# ── Prompt ────────────────────────────────────────────────────────────────────

_PLANNER_PROMPT = """You are a senior research strategist.
Given the following research topic, formulate:
1. A 3-sentence research plan covering overview, recent advances, and critical analysis.
2. Exactly 4 focused sub-questions that together fully cover the topic.
3. Exactly 4 concrete, Google-quality search queries (include year/domain terms).

Topic: {topic}

Return strictly a valid JSON object in the following format (no preamble, no markdown formatting):
{{
  "plan": "1. Overview sentence. 2. Recent advances sentence. 3. Critical analysis sentence.",
  "sub_questions": [
    "Sub-question 1",
    "Sub-question 2",
    "Sub-question 3",
    "Sub-question 4"
  ],
  "search_queries": [
    "Search query 1",
    "Search query 2",
    "Search query 3",
    "Search query 4"
  ]
}}
"""

def _fallback_queries(topic: str) -> List[str]:
    return [
        topic,
        f"{topic} overview state of the art 2024 2025",
        f"{topic} methodology and clinical or technical applications",
        f"{topic} challenges limitations benchmark analysis",
    ]

# ── Public node ───────────────────────────────────────────────────────────────

def run_planner_node(state: dict, llm=None) -> dict:
    """
    LangGraph node.
    Reads  : state['topic']
    Writes : state['plan'], state['sub_questions'],
             state['search_queries'], state['search_topic']
    """
    topic = state.get("topic", "").strip()
    if not topic:
        log.warning("PlannerAgent: empty topic, skipping.")
        return {
            **state,
            "plan": "",
            "sub_questions": [],
            "search_queries": [],
            "rewritten_queries": [],
            "search_topic": "",
        }

    try:
        prompt = _PLANNER_PROMPT.format(topic=topic)
        res = execute_with_fallback(f"[SYSTEM]\n{prompt}", tier="worker")
        content = res.content.strip()

        # Parse JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            plan = data.get("plan", "")
            sub_questions = data.get("sub_questions", [])
            search_queries = data.get("search_queries", [])
            
            if not isinstance(sub_questions, list) or not sub_questions:
                sub_questions = [topic]
            if not isinstance(search_queries, list) or not search_queries:
                search_queries = _fallback_queries(topic)

            primary_query = search_queries[0]
            log.info(
                "PlannerAgent (via %s): plan=%d chars | %d sub-questions | %d queries | primary=%r",
                res.provider_used,
                len(plan),
                len(sub_questions),
                len(search_queries),
                primary_query,
            )

            return {
                **state,
                "plan": plan,
                "sub_questions": sub_questions,
                "search_queries": search_queries,
                "rewritten_queries": search_queries,
                "search_topic": primary_query,
            }

    except Exception as exc:
        log.warning("PlannerAgent structured parse failed: %s. Using resilient query generator.", exc)

    fallback_qs = _fallback_queries(topic)
    return {
        **state,
        "plan": f"Comprehensive research investigation into {topic} covering current advances, practical applications, and emerging challenges.",
        "sub_questions": [
            f"What are the foundational principles of {topic}?",
            f"What are the latest breakthrough developments in {topic} (2024-2026)?",
            f"What are the real-world applications and benchmarks for {topic}?",
            f"What open challenges, limitations, and future directions exist for {topic}?"
        ],
        "search_queries": fallback_qs,
        "rewritten_queries": fallback_qs,
        "search_topic": topic,
    }