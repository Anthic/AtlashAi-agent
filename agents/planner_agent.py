import logging
from typing import List, Dict
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Structured output schema ──────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    plan: str = Field(description="3-sentence research plan covering overview, recent advances, and critical analysis")
    sub_questions: List[str] = Field(description="Exactly 4 focused sub-questions that together fully cover the topic")
    search_queries: List[str] = Field(description="One concrete Google-quality search query per sub-question, with year/domain terms")

# ── Prompt ────────────────────────────────────────────────────────────────────

_PLANNER_PROMPT = """\
You are a senior research strategist.

Given a research topic, produce:
1. A 3-sentence research plan covering overview, recent advances, and critical analysis.
2. Exactly 4 focused sub-questions that together fully cover the topic.
3. One concrete, Google-quality search query per sub-question (include year/domain terms).

Topic: {topic}
"""

# ── Public node ───────────────────────────────────────────────────────────────

def run_planner_node(state: dict, llm) -> dict:
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
        structured_llm = llm.with_structured_output(PlannerOutput)

        prompt = _PLANNER_PROMPT.format(topic=topic)
        parsed: PlannerOutput = structured_llm.invoke(prompt)

        # Fallback: if LLM returned empty lists
        if not parsed.search_queries:
            parsed.search_queries = [topic]
        if not parsed.sub_questions:
            parsed.sub_questions = [topic]

        primary_query = parsed.search_queries[0]

        log.info(
            "PlannerAgent: plan=%d chars | %d sub-questions | %d queries | primary=%r",
            len(parsed.plan),
            len(parsed.sub_questions),
            len(parsed.search_queries),
            primary_query,
        )

        return {
            **state,
            "plan": parsed.plan,
            "sub_questions": parsed.sub_questions,
            "search_queries": parsed.search_queries,
           
            "rewritten_queries": parsed.search_queries,
            "search_topic": primary_query,
        }

    except Exception as exc:
        log.exception("PlannerAgent failed")
        return {
            **state,
            "plan": "",
            "sub_questions": [],
            "search_queries": [topic],
            "rewritten_queries": [topic],
            "search_topic": topic,
            "error": f"PlannerAgent failed: {exc}",
        }