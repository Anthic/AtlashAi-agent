# 🧪 Multi-Agent Research System - Complete Testing Guide

## Overview

Your project has multiple testing layers. This guide explains how to test each component and the full workflow.

---

## Test Structure

```
testing/
├── test_planner_node.py          ← Unit tests for Planner Agent
├── test_searcher_node.py         ← Unit tests for Searcher Agent
├── test_knowledge_graph.py       ← Unit tests for Knowledge Graph Agent
├── test_workflow_integration.py  ← NEW: Integration tests (state progression)
└── test_manual_workflow.py       ← NEW: Manual E2E test (with mock data)
```

---

## 🚀 Quick Start: Test the Full Workflow

### Option 1: Run Manual E2E Test (Fastest)
No setup needed - tests with mock data immediately:

```bash
# From MultiAgentPart directory
python testing/test_manual_workflow.py
```

**What it does:**
- ✓ Simulates all 9 pipeline stages
- ✓ Validates state at each step
- ✓ Checks retry logic
- ✓ Prints timing for each stage
- ✓ Saves report to `outputs/`

**Expected output:**
```
════════════════════════════════════════════════════════════════════════════════
► MULTI-AGENT RESEARCH SYSTEM - MANUAL WORKFLOW TEST
════════════════════════════════════════════════════════════════════════════════

[Step 1] Planner Agent - Decompose Topic into Plan
✓ Planner: All required fields present
ℹ Generated plan with 4 sub-questions
ℹ Created 4 search queries

[Step 2] Searcher Agent - Multi-Query Search
✓ Searcher: All required fields present
ℹ Found 8 verified URLs
...
```

**Timing:**
Shows how long each stage takes (with mock data):
```
Timing Breakdown:
  planner              :   0.00s (  0.1%)
  searcher             :   0.00s (  0.2%)
  reader               :   0.00s (  0.3%)
  ...
  TOTAL                :   0.10s
```

---

### Option 2: Run Integration Tests (With Assertions)

Tests state transitions and data consistency:

```bash
# Run all integration tests
pytest testing/test_workflow_integration.py -v

# Run specific test class
pytest testing/test_workflow_integration.py::TestFullPipelineFlow -v

# Run with detailed output
pytest testing/test_workflow_integration.py -v -s
```

**Key Test Classes:**

1. **TestFullPipelineFlow** - State progression through all stages
   ```python
   • test_full_pipeline_state_progression()      # Validates each stage adds data
   • test_state_immutability_through_stages()    # Ensures original data preserved
   • test_error_handling_in_state()              # Error field maintained
   ```

2. **TestPipelineEdgeCases** - Error scenarios
   ```python
   • test_empty_search_results_recovery()        # Handle no results gracefully
   • test_retry_loop_logic()                     # Verify retry mechanism
   • test_empty_topic_validation()               # Catch invalid input early
   ```

3. **TestDataConsistency** - Data integrity
   ```python
   • test_url_deduplication()                    # URLs deduplicated
   • test_knowledge_graph_structure()            # Valid KG format
   ```

4. **TestPerformanceMetrics** - Timing tracking
   ```python
   • test_total_time_tracking()                  # Total execution time
   • test_stage_time_breakdown()                 # Per-stage timing
   ```

---

## 🔧 Individual Agent Tests

### Test Planner Agent
```bash
pytest testing/test_planner_node.py -v

# Output:
# test_planner_returns_all_fields PASSED
# test_planner_empty_topic_skips_llm PASSED
# test_planner_fallback_when_empty_queries PASSED
# test_planner_handles_llm_exception PASSED
```

**Tests:**
- ✓ Returns plan, sub_questions, search_queries
- ✓ Handles empty topic (skips LLM)
- ✓ Fallback when LLM returns empty lists
- ✓ Graceful error handling

---

### Test Searcher Agent
```bash
pytest testing/test_searcher_node.py -v

# Output:
# test_searcher_runs_all_queries PASSED
# test_searcher_restores_original_topic PASSED
# test_searcher_deduplicates_urls PASSED
```

**Tests:**
- ✓ Runs all planner queries
- ✓ Preserves original topic
- ✓ Deduplicates URLs across searches

---

### Test Knowledge Graph Agent
```bash
pytest testing/test_knowledge_graph.py -v
```

---

## 📊 Test All Agents At Once

```bash
# Run entire test suite
pytest testing/ -v

# Run with coverage report
pytest testing/ --cov=agents --cov=pipeline --cov-report=html

# Open coverage report
start htmlcov/index.html
```

---

## 🎯 Specific Testing Scenarios

### Scenario 1: Test with Real Config (Still Mock LLM)

Create `testing/test_real_config.py`:

```python
"""Test with real pipeline config but mock LLM"""
import pytest
from pipeline.runner import build_research_graph
from unittest.mock import MagicMock


def test_real_pipeline_with_mock_llm():
    """Build real graph but mock the LLM"""
    
    # Mock LLM
    mock_llm = MagicMock()
    
    # Build real graph
    graph = build_research_graph(mock_llm)
    
    # Test graph structure
    assert graph is not None
    assert hasattr(graph, 'invoke')
    
    # Run with test input
    result = graph.invoke({
        "topic": "test topic",
        "max_retries": 1,
    })
    
    assert result["topic"] == "test topic"
    assert "report" in result
```

Run it:
```bash
pytest testing/test_real_config.py -v
```

---

### Scenario 2: Test with Real Search (Single Query)

```python
"""Test single search query with real Tavily API"""
import pytest
from agents.search_agent import run_search_agent
import os


@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
def test_real_search():
    """Test real search with Tavily (requires API key)"""
    
    result = run_search_agent("quantum computing", MagicMock())
    
    assert "search_results" in result
    assert len(result["verified_urls"]) > 0
    assert isinstance(result["search_results"], list)


# Skip this test unless you set TAVILY_API_KEY:
# set TAVILY_API_KEY=your_key
# pytest testing/test_real_config.py::test_real_search -v
```

---

### Scenario 3: Full End-to-End with Backend API

Test the FastAPI backend workflow:

```bash
# Terminal 1: Start backend server
cd MultiAgentPart-Backend
npm run dev
# Starts on http://localhost:3000

# Terminal 2: Run E2E test
pytest testing/test_api_integration.py -v
```

Create `testing/test_api_integration.py`:

```python
"""Test backend API integration"""
import pytest
import requests
import asyncio


@pytest.mark.asyncio
async def test_research_endpoint():
    """Test POST /api/research endpoint"""
    
    # Create research request
    response = requests.post(
        "http://localhost:3000/api/research",
        json={"topic": "machine learning ethics"},
        timeout=120
    )
    
    assert response.status_code == 200
    result = response.json()
    
    assert "report" in result
    assert result["critique_score"] >= 0
    assert result["fact_check_score"] >= 0


async def test_stream_research():
    """Test streaming research output"""
    
    response = requests.get(
        "http://localhost:3000/api/research/stream",
        params={"topic": "quantum computing"},
        stream=True,
        timeout=120
    )
    
    chunks = []
    for line in response.iter_lines():
        if line:
            chunks.append(line.decode())
    
    assert len(chunks) > 0
```

---

## 📋 Testing Checklist

Use this to validate your complete system:

```
UNIT TESTS
  ☐ Planner Agent returns all fields
  ☐ Planner handles empty topic
  ☐ Planner handles LLM errors
  ☐ Searcher runs all queries
  ☐ Searcher deduplicates URLs
  ☐ Knowledge Graph builds valid structure

INTEGRATION TESTS
  ☐ State progresses through all stages
  ☐ Original data preserved through pipeline
  ☐ Error field maintained
  ☐ Empty results handled gracefully
  ☐ Retry logic works correctly
  ☐ URL deduplication works
  ☐ Knowledge graph is valid JSON

E2E TESTS
  ☐ Manual workflow completes (test_manual_workflow.py)
  ☐ Pipeline produces valid report
  ☐ Fact check score is valid (0-1)
  ☐ Critique score is valid (0-10)
  ☐ All URLs processed without errors
  ☐ Total time is reasonable

API TESTS
  ☐ /api/research endpoint returns report
  ☐ /api/research/stream streams output
  ☐ Error handling on invalid input
  ☐ Rate limiting works

PERFORMANCE
  ☐ Planner: < 5 seconds
  ☐ Searcher: < 30 seconds (4 queries)
  ☐ Reader: < 20 seconds
  ☐ Total pipeline: < 120 seconds
```

---

## 🔍 Debugging Failed Tests

### Check Log Files
```bash
# View test output with timestamps
pytest testing/ -v --tb=long > test_results.log
cat test_results.log

# View captured print statements
pytest testing/ -v -s
```

### Check State at Each Step
In your test, print state:

```python
def test_with_debug():
    state = create_initial_state("test topic")
    planner_output = mock_planner_stage(state)
    state.update(planner_output)
    
    # Print state at this point
    import json
    print(json.dumps({k: str(v)[:100] for k, v in state.items()}, indent=2))
```

### Verify Mock Data
```python
# Check planner output structure
def test_planner_structure():
    output = _mock_planner_output()
    
    assert "plan" in output
    assert "sub_questions" in output
    assert len(output["search_queries"]) > 0
    print(json.dumps(output, indent=2))
```

---

## 🎬 Running Tests in CI/CD

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: pytest testing/ -v
      
      - name: Run manual workflow
        run: python testing/test_manual_workflow.py
      
      - name: Upload coverage
        run: pytest testing/ --cov --cov-report=xml
```

### Run Locally (Simulate CI):
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run like CI would
pytest testing/ -v --cov --tb=short
```

---

## 📝 Writing Your Own Tests

### Template for Testing New Agents

```python
"""Test your new agent"""
from unittest.mock import MagicMock, patch
import pytest


class TestYourNewAgent:
    
    def test_agent_processes_input(self):
        """Test agent accepts input and produces output"""
        # Arrange
        mock_llm = MagicMock()
        input_state = {
            "topic": "test",
            "search_results": ["result1", "result2"],
        }
        
        # Act
        result = run_your_agent(input_state, mock_llm)
        
        # Assert
        assert result is not None
        assert "output_field" in result
        assert len(result["output_field"]) > 0
    
    
    def test_agent_handles_empty_input(self):
        """Test agent handles empty input gracefully"""
        # Arrange
        input_state = {"topic": "", "search_results": []}
        
        # Act
        result = run_your_agent(input_state, MagicMock())
        
        # Assert
        assert result is not None  # Should not crash
        assert "error" in result or "output_field" in result
    
    
    def test_agent_error_handling(self):
        """Test agent handles LLM errors"""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API Error")
        
        # Act & Assert
        with pytest.raises(RuntimeError):
            run_your_agent({"topic": "test"}, mock_llm)
```

---

## ✅ Next Steps

1. **Run manual workflow test:**
   ```bash
   python testing/test_manual_workflow.py
   ```

2. **Run integration tests:**
   ```bash
   pytest testing/test_workflow_integration.py -v
   ```

3. **Run all existing tests:**
   ```bash
   pytest testing/ -v
   ```

4. **Check coverage:**
   ```bash
   pytest testing/ --cov=agents --cov=pipeline --cov-report=html
   ```

5. **Fix any failures** and iterate

6. **Add real API tests** when you integrate actual services

---

## 📞 Support

If tests fail, check:
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Python 3.8+: `python --version`
- [ ] Tests can find modules: Add parent dir to path
- [ ] Mock data is valid: Print output to verify
- [ ] State matches expected schema: Check ResearchState

Good luck! 🚀
