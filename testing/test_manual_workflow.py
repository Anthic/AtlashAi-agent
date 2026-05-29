"""
test_manual_workflow.py
────────────────────────────────────────────────────────────
Manual end-to-end workflow test. Run this to test the complete pipeline
without external API calls (uses mocked data).

USAGE:
  python testing/test_manual_workflow.py
  
This script:
1. Creates mock state
2. Simulates each pipeline stage
3. Validates state at each step
4. Reports timing and results
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import ResearchState


# ──────────────────────────────────────────────────────────────────────────────
# ANSI Colors for console output
# ──────────────────────────────────────────────────────────────────────────────

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}► {title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")


def print_step(step_num: int, title: str):
    print(f"\n{Colors.CYAN}[Step {step_num}] {title}{Colors.END}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_info(msg: str):
    print(f"{Colors.CYAN}ℹ {msg}{Colors.END}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


# ──────────────────────────────────────────────────────────────────────────────
# Mock Data Generators
# ──────────────────────────────────────────────────────────────────────────────

def create_initial_state(topic: str) -> Dict[str, Any]:
    """Create initial research state"""
    return {
        "topic": topic,
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


def mock_planner_stage(state: Dict) -> Dict:
    """Simulate Planner Agent"""
    return {
        "plan": """# Research Plan

## Part 1: Foundations
- Understanding core concepts
- Historical context
- Current state of art

## Part 2: Deep Dive
- Technical implementation
- Case studies
- Comparative analysis

## Part 3: Synthesis
- Key findings
- Best practices
- Future directions""",
        "sub_questions": [
            "What are the foundational concepts in this domain?",
            "How has this field evolved over time?",
            "What are the current best practices?",
            "What are the emerging trends and challenges?",
        ],
        "search_queries": [
            f"{state['topic']} fundamentals concepts",
            f"{state['topic']} best practices 2024",
            f"{state['topic']} case studies applications",
            f"{state['topic']} challenges emerging trends",
        ],
    }


def mock_searcher_stage(state: Dict) -> Dict:
    """Simulate Searcher Agent (runs multiple search queries)"""
    results = []
    urls = []

    for i, query in enumerate(state.get("search_queries", []), 1):
        results.append(f"Search result {i} for '{query}': Lorem ipsum dolor sit amet, consectetur adipiscing elit...")
        urls.append(f"https://example-research.com/article-{i}")
        urls.append(f"https://academic-source.org/paper-{i}")

    return {
        "search_results": results,
        "verified_urls": list(dict.fromkeys(urls)),  # Deduplicate
        "urls": list(dict.fromkeys(urls)),
    }


def mock_reader_stage(state: Dict) -> Dict:
    """Simulate Reader Agent (scrape and verify URLs)"""
    content_chunks = []
    for url in state.get("verified_urls", [])[:3]:  # Limit to 3 for demo
        content_chunks.append(f"""
--- Content from {url} ---
This is sample scraped content from the verified URL.
It contains relevant information about the topic: {state['topic']}

Key points:
1. Important finding 1
2. Important finding 2
3. Important finding 3

Sources and references mentioned...
""")

    return {
        "scraped_content": "\n".join(content_chunks),
    }


def mock_summarizer_stage(state: Dict) -> Dict:
    """Simulate Summarizer Agent"""
    return {
        "summarized_content": f"""# Summary of Research on: {state['topic']}

## Key Findings
1. First major insight from research
2. Second major insight from research
3. Third major insight from research

## Common Themes
- Theme A appears across multiple sources
- Theme B is emphasized in recent research
- Theme C represents an emerging perspective

## Notable Examples
- Example 1: Demonstrates application of theory
- Example 2: Shows real-world implementation
- Example 3: Illustrates best practices

## Research Gaps
- Area needing further investigation
- Understudied aspect of the topic
- Emerging challenges

Sources: {len(state.get('verified_urls', []))} verified sources analyzed
""",
    }


def mock_rag_stage(state: Dict) -> Dict:
    """Simulate RAG Retrieval"""
    return {
        "rag_context": f"""[RETRIEVAL 1] Fundamental concept: {state['topic']} refers to...
[RETRIEVAL 2] Historical context: The field emerged when...
[RETRIEVAL 3] Current best practice: Leading experts recommend...
[RETRIEVAL 4] Statistical evidence: Research shows that...
[RETRIEVAL 5] Industry standards: Companies implement... 
[RETRIEVAL 6] Theoretical framework: The foundation rests on...
[RETRIEVAL 7] Practical applications: Real-world usage includes...
""",
    }


def mock_writer_stage(state: Dict) -> Dict:
    """Simulate Writer Agent"""
    return {
        "report": f"""# Comprehensive Research Report: {state['topic']}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report synthesizes research on {state['topic']} based on {len(state.get('verified_urls', []))} verified sources. 
The analysis follows the structured research plan outlined in Section 1.

## 1. Research Plan Overview

{state.get('plan', 'No plan generated')}

## 2. Foundations and Core Concepts

{state['topic']} can be understood through several fundamental frameworks:
- First framework: Description of framework 1
- Second framework: Description of framework 2
- Third framework: Description of framework 3

## 3. Literature Review

Based on the research conducted:

### 3.1 Key Themes
- Theme 1: Multiple sources support this finding
- Theme 2: Emerging consensus on this aspect
- Theme 3: Divergent views on this perspective

### 3.2 Notable Case Studies
- Case A: Demonstrates successful implementation
- Case B: Shows challenges and solutions
- Case C: Illustrates future possibilities

## 4. Analysis and Findings

{state.get('summarized_content', 'Summary not available')}

## 5. Best Practices and Recommendations

1. Recommendation 1: Based on research evidence
2. Recommendation 2: Supported by multiple sources
3. Recommendation 3: Emerging best practice

## 6. Challenges and Future Directions

### Current Challenges
- Challenge A: Requires further research
- Challenge B: Industry-wide concern
- Challenge C: Technical limitation

### Future Research Directions
- Direction 1: Promising new area
- Direction 2: Interdisciplinary opportunity
- Direction 3: Practical implementation frontier

## 7. Conclusion

The research on {state['topic']} reveals a complex and evolving landscape with clear best practices 
emerging alongside new challenges. Continued research and implementation focus is recommended.

---
*Report generated by Multi-Agent Research System*
*Sources analyzed: {len(state.get('verified_urls', []))}*
*Query decomposition: {len(state.get('search_queries', []))} search queries*
""",
    }


def mock_knowledge_graph_stage(state: Dict) -> Dict:
    """Simulate Knowledge Graph Agent"""
    return {
        "knowledge_graph": {
            "nodes": [
                {"id": "topic", "label": state['topic'], "type": "root"},
                {"id": "concept1", "label": "Core Concept 1", "type": "concept"},
                {"id": "concept2", "label": "Core Concept 2", "type": "concept"},
                {"id": "practice1", "label": "Best Practice 1", "type": "practice"},
                {"id": "practice2", "label": "Best Practice 2", "type": "practice"},
            ],
            "edges": [
                {"source": "topic", "target": "concept1", "relation": "consists_of"},
                {"source": "topic", "target": "concept2", "relation": "consists_of"},
                {"source": "concept1", "target": "practice1", "relation": "leads_to"},
                {"source": "concept2", "target": "practice2", "relation": "enables"},
                {"source": "practice1", "target": "practice2", "relation": "complements"},
            ],
        },
        "knowledge_graph_md": """# Knowledge Graph

## Nodes
- Root: {topic}
- Concepts: Core Concept 1, Core Concept 2
- Practices: Best Practice 1, Best Practice 2

## Relationships
- Topic → Concept 1 (consists_of)
- Topic → Concept 2 (consists_of)
- Concept 1 → Practice 1 (leads_to)
- Concept 2 → Practice 2 (enables)
- Practice 1 ↔ Practice 2 (complements)
""".format(topic=state['topic']),
    }


def mock_fact_check_stage(state: Dict) -> Dict:
    """Simulate Fact-Check Agent"""
    total_claims = 7
    verified_claims = 6
    needs_review = 1

    return {
        "fact_check_result": f"""# Fact Check Report

Total Claims Checked: {total_claims}
✓ Verified: {verified_claims}
? Needs Review: {needs_review}
✗ Unverified: 0

## Verified Claims
✓ Claim 1: Confirmed by multiple sources
✓ Claim 2: Academic consensus
✓ Claim 3: Industry standard
✓ Claim 4: Statistical evidence
✓ Claim 5: Expert opinion
✓ Claim 6: Documented case study

## Claims Needing Review
? Claim A: Requires additional source verification

Accuracy Score: {(verified_claims / total_claims) * 100:.1f}%
""",
        "fact_check_score": 0.87,
    }


def mock_critic_stage(state: Dict) -> Dict:
    """Simulate Critic Agent"""
    return {
        "critique": """# Critique and Quality Assessment

## Report Quality Score: 8.2/10

### Strengths
+ Clear structure following research plan
+ Well-sourced from verified academic sources
+ Good balance of theory and practical applications
+ Accessible writing for broad audience
+ Comprehensive coverage of topic

### Areas for Improvement
- Could include more recent 2024 data
- Some sections could benefit from statistical evidence
- Limited discussion of implementation barriers
- Recommendations could be more specific

### Recommendations
→ Add quantitative metrics where available
→ Include timeline of developments
→ Expand on practical implementation guidance
→ Add comparison with competing approaches

### Recommendation
READY FOR PUBLICATION with minor enhancements.
Score indicates high-quality research report suitable for stakeholder distribution.
""",
        "critique_score": 8,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ──────────────────────────────────────────────────────────────────────────────

def validate_stage_output(stage_name: str, state: Dict, required_fields: list) -> bool:
    """Validate that required fields are populated"""
    missing = [f for f in required_fields if not state.get(f)]
    
    if missing:
        print_warning(f"{stage_name}: Missing fields: {missing}")
        return False
    
    print_success(f"{stage_name}: All required fields present")
    return True


def print_state_snapshot(stage_name: str, state: Dict, show_fields: list):
    """Print snapshot of state at each stage"""
    print_info(f"\nState snapshot after {stage_name}:")
    for field in show_fields:
        value = state.get(field, "")
        if isinstance(value, list):
            print(f"  • {field}: {len(value)} items")
        elif isinstance(value, str):
            preview = value[:80] + "..." if len(value) > 80 else value
            print(f"  • {field}: {preview}")
        elif isinstance(value, dict):
            print(f"  • {field}: {list(value.keys())}")
        else:
            print(f"  • {field}: {value}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Workflow Test
# ──────────────────────────────────────────────────────────────────────────────

def run_manual_workflow_test():
    """Execute complete workflow with timing and validation"""
    
    print_section("MULTI-AGENT RESEARCH SYSTEM - MANUAL WORKFLOW TEST")
    print(f"{Colors.CYAN}Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")

    # Configuration
    test_topic = "Ai impact on mental health"
    max_retries = 2
    retry_count = 0

    # Timing tracker
    stage_times = {}
    
    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 0: Initialize
    # ──────────────────────────────────────────────────────────────────────────
    print_step(0, "Initialize State")
    state = create_initial_state(test_topic)
    print_success(f"Initial state created for topic: '{test_topic}'")
    
    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: Planner
    # ──────────────────────────────────────────────────────────────────────────
    print_step(1, "Planner Agent - Decompose Topic into Plan")
    start = time.time()
    planner_output = mock_planner_stage(state)
    state.update(planner_output)
    stage_times["planner"] = time.time() - start
    
    validate_stage_output("Planner", state, ["plan", "sub_questions", "search_queries"])
    print_info(f"Generated plan with {len(state['sub_questions'])} sub-questions")
    print_info(f"Created {len(state['search_queries'])} search queries")
    print_state_snapshot("Planner", state, ["plan", "search_queries"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: Searcher
    # ──────────────────────────────────────────────────────────────────────────
    print_step(2, "Searcher Agent - Multi-Query Search")
    start = time.time()
    searcher_output = mock_searcher_stage(state)
    state.update(searcher_output)
    stage_times["searcher"] = time.time() - start
    
    validate_stage_output("Searcher", state, ["verified_urls", "search_results"])
    print_info(f"Found {len(state['verified_urls'])} verified URLs")
    print_info(f"Aggregated {len(state['search_results'])} search results")
    print_state_snapshot("Searcher", state, ["verified_urls"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: Reader
    # ──────────────────────────────────────────────────────────────────────────
    print_step(3, "Reader Agent - Scrape & Verify URLs")
    start = time.time()
    reader_output = mock_reader_stage(state)
    state.update(reader_output)
    stage_times["reader"] = time.time() - start
    
    validate_stage_output("Reader", state, ["scraped_content"])
    print_info(f"Scraped {len(state['verified_urls'])} URLs successfully")
    print_info(f"Total content length: {len(state['scraped_content'])} characters")
    print_state_snapshot("Reader", state, ["scraped_content"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: Summarizer
    # ──────────────────────────────────────────────────────────────────────────
    print_step(4, "Summarizer Agent - Condense Content")
    start = time.time()
    summarizer_output = mock_summarizer_stage(state)
    state.update(summarizer_output)
    stage_times["summarizer"] = time.time() - start
    
    validate_stage_output("Summarizer", state, ["summarized_content"])
    print_info(f"Condensed to {len(state['summarized_content'])} characters")
    print_state_snapshot("Summarizer", state, ["summarized_content"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: RAG
    # ──────────────────────────────────────────────────────────────────────────
    print_step(5, "RAG Node - Vector Similarity Retrieval")
    start = time.time()
    rag_output = mock_rag_stage(state)
    state.update(rag_output)
    stage_times["rag"] = time.time() - start
    
    validate_stage_output("RAG", state, ["rag_context"])
    print_info(f"Retrieved contextual information: {len(state['rag_context'])} characters")
    print_state_snapshot("RAG", state, ["rag_context"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 6: Knowledge Graph
    # ──────────────────────────────────────────────────────────────────────────
    print_step(6, "Knowledge Graph Agent - Extract Relationships")
    start = time.time()
    kg_output = mock_knowledge_graph_stage(state)
    state.update(kg_output)
    stage_times["knowledge_graph"] = time.time() - start
    
    validate_stage_output("Knowledge Graph", state, ["knowledge_graph", "knowledge_graph_md"])
    print_info(f"Built knowledge graph with {len(state['knowledge_graph'].get('nodes', []))} nodes")
    print_info(f"Created {len(state['knowledge_graph'].get('edges', []))} relationships")
    print_state_snapshot("Knowledge Graph", state, ["knowledge_graph_md"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 7: Writer
    # ──────────────────────────────────────────────────────────────────────────
    print_step(7, "Writer Agent - Generate Report")
    start = time.time()
    writer_output = mock_writer_stage(state)
    state.update(writer_output)
    stage_times["writer"] = time.time() - start
    
    validate_stage_output("Writer", state, ["report"])
    print_info(f"Generated report: {len(state['report'])} characters")
    print_state_snapshot("Writer", state, ["report"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 8: Fact Check
    # ──────────────────────────────────────────────────────────────────────────
    print_step(8, "Fact Check Agent - Verify Claims")
    start = time.time()
    fc_output = mock_fact_check_stage(state)
    state.update(fc_output)
    stage_times["fact_check"] = time.time() - start
    
    validate_stage_output("Fact Check", state, ["fact_check_result", "fact_check_score"])
    print_info(f"Fact check score: {state['fact_check_score']:.2f}")
    print_state_snapshot("Fact Check", state, ["fact_check_score"])

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 9: Critic
    # ──────────────────────────────────────────────────────────────────────────
    print_step(9, "Critic Agent - Quality Assessment")
    start = time.time()
    critic_output = mock_critic_stage(state)
    state.update(critic_output)
    stage_times["critic"] = time.time() - start
    
    validate_stage_output("Critic", state, ["critique", "critique_score"])
    print_info(f"Critique score: {state['critique_score']}/10")
    print_state_snapshot("Critic", state, ["critique_score"])

    # ──────────────────────────────────────────────────────────────────────────
    # CONDITIONAL RETRY LOGIC
    # ──────────────────────────────────────────────────────────────────────────
    print_step(10, "Check Critique Score for Retry Logic")
    
    if state['critique_score'] < 8 and state['retry_count'] < state['max_retries']:
        print_warning(f"Score {state['critique_score']} < 8. Triggering retry...")
        state['retry_count'] += 1
        print_info(f"Retry {state['retry_count']}/{state['max_retries']}: Would re-run Writer → Critic loop")
        # In real pipeline, would jump back to Writer stage
    else:
        print_success(f"Score {state['critique_score']} is acceptable or max retries reached")
        print_success("Proceeding to finalization")

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL RESULTS
    # ──────────────────────────────────────────────────────────────────────────
    print_section("WORKFLOW EXECUTION COMPLETED")

    print(f"\n{Colors.BOLD}Timing Breakdown:{Colors.END}")
    total_time = sum(stage_times.values())
    
    if total_time > 0:
        for stage, elapsed in stage_times.items():
            percentage = (elapsed / total_time) * 100
            print(f"  {stage:20s}: {elapsed:7.2f}s ({percentage:5.1f}%)")
    else:
        for stage, elapsed in stage_times.items():
            print(f"  {stage:20s}: {elapsed:7.2f}s (< 0.1%)")
    
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':20s}: {total_time:7.2f}s (with mock data)")

    print(f"\n{Colors.BOLD}Final State Summary:{Colors.END}")
    print(f"  Topic: {state['topic']}")
    print(f"  Plan Generated: {'Yes' if state['plan'] else 'No'}")
    print(f"  Search Queries: {len(state['search_queries'])}")
    print(f"  URLs Found: {len(state['verified_urls'])}")
    print(f"  Report Length: {len(state['report'])} chars")
    print(f"  Knowledge Graph Nodes: {len(state['knowledge_graph'].get('nodes', []))}")
    print(f"  Fact Check Score: {state['fact_check_score']:.2f}")
    print(f"  Critique Score: {state['critique_score']}/10")
    print(f"  Retry Count: {state['retry_count']}/{state['max_retries']}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All stages completed successfully!{Colors.END}")
    
    # Save report
    report_path = Path(__file__).parent.parent / "outputs" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(state['report'])
    print(f"\n{Colors.GREEN}Report saved to: {report_path}{Colors.END}")

    print(f"\n{Colors.CYAN}Test End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")


if __name__ == "__main__":
    run_manual_workflow_test()
