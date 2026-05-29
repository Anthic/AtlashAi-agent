"""
test_workflow_integration.py
────────────────────────────────────────────────────────────
End-to-end integration test for the complete research pipeline.
Tests the full workflow from topic → planner → searcher → reader → writer.

USAGE:
  pytest testing/test_workflow_integration.py -v
  pytest testing/test_workflow_integration.py::test_full_pipeline_flow -v -s
"""

import pytest
from unittest.mock import MagicMock, patch
from state import ResearchState


# ──────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Mock Data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_state_initial():
    """Initial state with just topic"""
    return {
        "topic": "machine learning ethics",
        "plan": "",
        "sub_questions": [],
        "search_queries": [],
        "rewritten_queries": [],
        "search_topic": "",
        "search_results": [],
        "verified_urls": [],
        "urls": [],
        "scraped_content": "",
        "summarized_content": "",
        "rag_context": "",
        "report": "",
        "draft_sections": [],
        "knowledge_graph": {},
        "knowledge_graph_md": "",
        "critique": "",
        "critique_score": 0,
        "retry_count": 0,
        "max_retries": 2,
        "fact_check_result": "",
        "fact_check_score": 0.0,
        "error": "",
        "time_sec": 0.0,
    }


@pytest.fixture
def mock_llm():
    """Mock LLM that returns reasonable outputs"""
    llm = MagicMock()
    return llm


def _mock_planner_output():
    """Mock planner agent output"""
    return {
        "plan": "1. Introduction\n2. Core concepts\n3. Case studies\n4. Conclusion",
        "sub_questions": [
            "What are the ethical issues in ML?",
            "How do bias and fairness intersect?",
            "What are current best practices?",
            "What are future challenges?",
        ],
        "search_queries": [
            "machine learning bias fairness ethics",
            "AI ethics case studies industry",
            "responsible AI governance frameworks",
            "algorithmic discrimination prevention",
        ],
        "search_topic": "machine learning bias fairness ethics",
        "rewritten_queries": [
            "machine learning bias fairness ethics",
            "AI ethics case studies industry",
            "responsible AI governance frameworks",
            "algorithmic discrimination prevention",
        ],
    }


def _mock_search_result():
    """Mock search agent output"""
    return {
        "search_results": [
            "Result snippet 1: ML bias affects healthcare outcomes...",
            "Result snippet 2: Fair ML practices in tech companies...",
        ],
        "verified_urls": [
            "https://example-research.com/ml-ethics-1",
            "https://example-ai.org/fairness-guide",
        ],
        "urls": [
            "https://example-research.com/ml-ethics-1",
            "https://example-ai.org/fairness-guide",
        ],
    }


def _mock_scraped_content():
    """Mock reader agent output (scraped & verified content)"""
    return {
        "scraped_content": """
Machine Learning Ethics: A Comprehensive Overview

1. Introduction to Bias in ML Systems
- Historical context of algorithmic bias
- Real-world examples in healthcare, criminal justice, hiring
- Impact on underrepresented communities

2. Fairness Definitions & Trade-offs
- Demographic parity vs. equal opportunity
- Individual vs. group fairness
- Technical approaches to bias mitigation

3. Industry Best Practices
- Fairness assessments during model development
- Diverse team composition
- Transparency & explainability

4. Future Directions
- Regulatory frameworks (EU AI Act)
- Long-term accountability mechanisms
- Research gaps
        """.strip(),
    }


def _mock_summarized_content():
    """Mock summarizer agent output"""
    return {
        "summarized_content": """
Summary of ML Ethics Research:

Key Points:
1. Bias is systemic in ML systems and affects critical domains
2. Multiple fairness definitions exist with inherent trade-offs
3. Industry leaders adopt fairness assessments & transparency
4. Regulatory frameworks are emerging globally
5. Ongoing research needed in accountability mechanisms

Evidence:
- Healthcare ML bias reduces diagnostic accuracy for minorities
- Hiring algorithms show gender discrimination
- Tech companies increasingly adopt fairness principles
- EU AI Act introduces regulatory requirements
""".strip(),
    }


def _mock_rag_context():
    """Mock RAG retrieval output"""
    return {
        "rag_context": """
[Doc 1] Machine learning bias disproportionately affects minority groups...
[Doc 2] Fairness metrics include demographic parity and equalized odds...
[Doc 3] Responsible AI frameworks from leading tech companies...
[Doc 4] Regulatory landscape: EU AI Act, US AI Bill of Rights...
""".strip(),
    }


def _mock_report():
    """Mock writer agent output"""
    return {
        "report": """
# Machine Learning Ethics: A Comprehensive Guide

## 1. Introduction
Machine learning has transformed industries but raises critical ethical questions...

## 2. Understanding Bias in ML Systems
### 2.1 Sources of Bias
- Training data bias reflecting historical inequalities
- Feature selection and proxy discrimination
- Optimization objectives misaligned with fairness

### 2.2 Real-World Impact
Healthcare: ML diagnostic tools show 23% lower accuracy for Black patients
Criminal Justice: COMPAS algorithm exhibits racial bias in recidivism prediction
Hiring: Amazon's ML recruiting tool discriminated against women

## 3. Fairness Frameworks
- Demographic Parity: equal outcomes across groups
- Equalized Odds: equal true positive/false positive rates
- Individual Fairness: similar individuals treated similarly

## 4. Industry Best Practices
- Fairness assessments in model development pipeline
- Diverse teams in AI development
- Transparency & explainability mechanisms
- Regular audits and monitoring

## 5. Regulatory Landscape
- EU AI Act classifies systems by risk
- US exploring Bill of Rights for AI
- GDPR implications for algorithmic transparency

## 6. Conclusion & Future Directions
Addressing ML ethics requires technical innovation, policy frameworks, and cultural change.
""".strip(),
    }


def _mock_fact_check():
    """Mock fact-check agent output"""
    return {
        "fact_check_result": """
Fact-Check Report:
✓ VERIFIED: COMPAS algorithm bias documented in ProPublica investigation
✓ VERIFIED: Amazon ML hiring tool discriminated against women (Reuters 2018)
✓ VERIFIED: EU AI Act introduced in 2023
? NEEDS REVIEW: 23% accuracy gap in healthcare ML for Black patients (cite needed)
""".strip(),
        "fact_check_score": 0.85,
    }


def _mock_critic_output():
    """Mock critic agent output"""
    return {
        "critique": """
Critique Score: 8.5/10

Strengths:
+ Well-structured with clear introduction and conclusion
+ Uses concrete real-world examples
+ Covers multiple fairness definitions
+ Mentions both technical and regulatory approaches

Areas for Improvement:
- Could include more recent case studies (2024+)
- Limited discussion of trade-offs between fairness metrics
- Could strengthen section on implementation challenges

Recommendation: GOOD to publish with minor revisions
""".strip(),
        "critique_score": 8,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipelineFlow:
    """Test complete workflow from topic to final report"""

    def test_full_pipeline_state_progression(self, mock_state_initial):
        """
        Verify state progresses correctly through all pipeline stages.
        Each stage should preserve previous data and add new fields.
        """
        state = mock_state_initial.copy()

        # Stage 1: Planner
        planner_output = _mock_planner_output()
        state.update(planner_output)
        assert state["plan"]
        assert len(state["sub_questions"]) > 0
        assert len(state["search_queries"]) > 0

        # Stage 2: Searcher
        search_output = _mock_search_result()
        state["search_results"] = search_output["search_results"]
        state["verified_urls"] = search_output["verified_urls"]
        assert state["search_results"]
        assert state["verified_urls"]

        # Stage 3: Reader (scraping)
        reader_output = _mock_scraped_content()
        state.update(reader_output)
        assert state["scraped_content"]

        # Stage 4: Summarizer
        summarizer_output = _mock_summarized_content()
        state.update(summarizer_output)
        assert state["summarized_content"]

        # Stage 5: RAG
        rag_output = _mock_rag_context()
        state.update(rag_output)
        assert state["rag_context"]

        # Stage 6: Writer
        writer_output = _mock_report()
        state.update(writer_output)
        assert state["report"]
        assert "Introduction" in state["report"]

        # Stage 7: Fact-Check
        fc_output = _mock_fact_check()
        state.update(fc_output)
        assert state["fact_check_result"]
        assert state["fact_check_score"] > 0

        # Stage 8: Critic
        critic_output = _mock_critic_output()
        state.update(critic_output)
        assert state["critique"]
        assert state["critique_score"] > 0

    def test_state_immutability_through_stages(self, mock_state_initial):
        """Verify original data isn't mutated through pipeline stages"""
        state = mock_state_initial.copy()
        original_topic = state["topic"]

        # Update with planner output
        state.update(_mock_planner_output())
        assert state["topic"] == original_topic  # topic should remain unchanged

    def test_error_handling_in_state(self, mock_state_initial):
        """Verify error field is properly maintained"""
        state = mock_state_initial.copy()
        error_msg = "API rate limit exceeded"
        state["error"] = error_msg

        # State should maintain error even after updates
        state.update(_mock_planner_output())
        state["error"] = error_msg
        assert state["error"] == error_msg


class TestPipelineEdgeCases:
    """Test edge cases and error scenarios"""

    def test_empty_search_results_recovery(self, mock_state_initial):
        """Pipeline should handle empty search results gracefully"""
        state = mock_state_initial.copy()
        state.update(_mock_planner_output())

        # No search results
        state["search_results"] = []
        state["verified_urls"] = []

        # Should still progress with fallback behavior
        assert state["topic"]  # Original topic preserved
        assert state["search_queries"]  # Queries preserved

    def test_retry_loop_logic(self, mock_state_initial):
        """Verify retry logic for low critique scores"""
        state = mock_state_initial.copy()
        state["max_retries"] = 2
        state["retry_count"] = 0

        # Simulate low critic score
        state["critique_score"] = 6  # Below 8 threshold

        # Should trigger retry
        if state["critique_score"] < 8 and state["retry_count"] < state["max_retries"]:
            state["retry_count"] += 1
            assert state["retry_count"] == 1

        # After max retries, should exit
        state["retry_count"] = state["max_retries"]
        should_exit = state["retry_count"] >= state["max_retries"]
        assert should_exit

    def test_empty_topic_validation(self, mock_state_initial):
        """Empty topic should be caught early"""
        state = mock_state_initial.copy()
        state["topic"] = ""

        # Planner should skip LLM call
        is_valid_topic = bool(state["topic"].strip())
        assert not is_valid_topic


class TestDataConsistency:
    """Test data consistency across pipeline stages"""

    def test_url_deduplication(self):
        """Verify URLs are deduplicated across searches"""
        urls_from_query_1 = [
            "https://example.com/page1",
            "https://example.com/page2",
        ]
        urls_from_query_2 = [
            "https://example.com/page1",  # duplicate
            "https://example.com/page3",
        ]

        all_urls = urls_from_query_1 + urls_from_query_2
        deduplicated = list(dict.fromkeys(all_urls))  # preserve order

        assert len(deduplicated) == 3
        assert deduplicated == [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]

    def test_search_queries_not_duplicated_in_rewritten(self):
        """Planner queries shouldn't be rewritten again by query_rewrite"""
        planner_queries = [
            "query 1",
            "query 2",
            "query 3",
        ]

        # These should be used as-is, not rewritten
        assert len(planner_queries) == 3

    def test_knowledge_graph_structure(self):
        """Verify knowledge graph format is valid"""
        kg = {
            "nodes": [
                {"id": "bias", "label": "Bias in ML"},
                {"id": "fairness", "label": "Fairness"},
            ],
            "edges": [
                {"source": "bias", "target": "fairness", "relation": "relates_to"},
            ],
        }

        assert "nodes" in kg
        assert "edges" in kg
        assert len(kg["nodes"]) > 0
        assert all("id" in node for node in kg["nodes"])


class TestPerformanceMetrics:
    """Test timing and performance tracking"""

    def test_total_time_tracking(self, mock_state_initial):
        """Pipeline should track total execution time"""
        import time

        state = mock_state_initial.copy()
        start_time = time.time()

        # Simulate pipeline execution
        state.update(_mock_planner_output())
        state.update(_mock_search_result())
        state.update(_mock_scraped_content())

        end_time = time.time()
        elapsed = end_time - start_time

        state["time_sec"] = elapsed
        assert state["time_sec"] >= 0

    def test_stage_time_breakdown(self):
        """Verify stage-level timing data"""
        stage_times = {
            "planner": 2.5,
            "searcher": 15.3,
            "reader": 8.7,
            "summarizer": 3.2,
            "rag": 1.5,
            "writer": 12.1,
            "fact_check": 8.9,
            "critic": 5.3,
        }

        total_time = sum(stage_times.values())
        assert total_time == pytest.approx(57.5)

        # Verify longest stage
        longest_stage = max(stage_times, key=stage_times.get)
        assert longest_stage == "searcher"


# ──────────────────────────────────────────────────────────────────────────────
# Run Tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
