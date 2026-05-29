# 🧪 Testing Directory

This directory contains all tests for the Multi-Agent Research System.

## 📁 Files

### Test Files
- **test_planner_node.py** - Unit tests for Planner Agent
- **test_searcher_node.py** - Unit tests for Searcher Agent  
- **test_knowledge_graph.py** - Unit tests for Knowledge Graph Agent
- **test_workflow_integration.py** - Integration tests for full workflow
- **test_manual_workflow.py** - Manual E2E test with mock data

### Configuration
- **conftest.py** - Pytest configuration and shared fixtures
- **pytest.ini** - Pytest settings
- **TESTING_GUIDE.md** - Complete testing documentation

---

## ⚡ Quick Commands

### Run All Tests
```bash
pytest
```

### Run Manual Workflow Test (No Setup)
```bash
python test_manual_workflow.py
```

### Run Integration Tests Only
```bash
pytest test_workflow_integration.py -v
```

### Run with Coverage
```bash
pytest --cov=agents --cov=pipeline --cov-report=html
```

### Run Only Unit Tests (Skip Integration)
```bash
pytest -m "not integration"
```

### Run Only Slow Tests
```bash
pytest -m "slow" -v
```

---

## 🔧 Test Markers

Tests are marked with pytest markers for easy filtering:

```bash
# Run only unit tests
pytest -m "unit"

# Run integration tests
pytest -m "integration"

# Skip tests requiring API keys
pytest -m "not requires_api"

# Run specific test
pytest test_workflow_integration.py::TestFullPipelineFlow::test_full_pipeline_state_progression -v
```

---

## 📊 Coverage Report

Generate HTML coverage report:

```bash
pytest --cov=agents --cov=pipeline --cov-report=html
open htmlcov/index.html
```

---

## 🚀 Running Tests from Different Locations

### From project root:
```bash
pytest MultiAgentPart/testing -v
```

### From testing directory:
```bash
cd testing
pytest
```

### From anywhere (if pytest.ini is in testing/):
```bash
pytest --rootdir=MultiAgentPart/testing
```

---

## 📝 Adding New Tests

1. Create file: `test_your_feature.py`
2. Write test class: `class TestYourFeature:`
3. Add test methods: `def test_something():`
4. Add markers if needed: `@pytest.mark.integration`
5. Run: `pytest test_your_feature.py -v`

Example:
```python
"""test_new_agent.py"""
import pytest
from agents.your_agent import run_your_agent

class TestNewAgent:
    """Tests for new agent"""
    
    @pytest.mark.unit
    def test_agent_basic_flow(self, mock_llm, sample_state):
        """Test agent basic functionality"""
        result = run_your_agent(sample_state, mock_llm)
        assert result is not None
    
    @pytest.mark.integration
    def test_agent_with_real_data(self):
        """Test agent with real data"""
        pass
```

---

## 🔍 Fixtures Available

From `conftest.py`:

- **mock_llm** - Mocked LLM model
- **mock_embeddings** - Mocked embeddings model
- **mock_vectorstore** - Mocked vector store
- **sample_state** - Sample research state
- **mock_search_results** - Sample search results
- **test_config** - Test configuration
- **various_topics** - Multiple topics for parametrized tests
- **critique_scores** - Various critique scores

Usage:
```python
def test_something(mock_llm, sample_state):
    result = run_agent(sample_state, mock_llm)
    assert result is not None
```

---

## 🐛 Debugging Tests

### View detailed output:
```bash
pytest -v -s test_file.py
```

### Stop on first failure:
```bash
pytest -x
```

### Run with print statements:
```bash
pytest -v -s
```

### Run with traceback:
```bash
pytest --tb=long
```

---

## 📈 Test Results

### Expected Results
- All unit tests: ✓ PASS
- Integration tests: ✓ PASS  
- E2E test: ✓ PASS
- Coverage: > 80%

### If Tests Fail
1. Check Python version: `python --version` (need 3.8+)
2. Check dependencies: `pip install -r ../requirements.txt`
3. Check imports: `python -c "from state import ResearchState"`
4. Read error message carefully
5. Check TESTING_GUIDE.md for debugging

---

## 🎯 Test Categories

### Unit Tests
- Individual agent functionality
- Error handling
- Edge cases

### Integration Tests  
- State progression through pipeline
- Data consistency
- Retry logic

### E2E Tests
- Full workflow execution
- Report generation
- Fact checking

### Performance Tests
- Stage timing
- Total execution time
- Resource usage

---

## 📞 Troubleshooting

### Import errors:
```python
# In test file, add:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Fixture not found:
```bash
# Ensure conftest.py is in testing/ directory
# Or add to pytest.ini:
# confcutdir = .
```

### Slow tests:
```bash
# Skip slow tests:
pytest -m "not slow"
```

### Failed setup:
```bash
# Check environment:
python -c "import pytest; print(pytest.__version__)"
python -c "from dotenv import load_dotenv; load_dotenv()"
```

---

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Mock Objects](https://docs.python.org/3/library/unittest.mock.html)
- [TESTING_GUIDE.md](./TESTING_GUIDE.md) - Full testing guide

---

## ✅ Pre-commit Testing

Run before committing:

```bash
# Quick smoke test
python testing/test_manual_workflow.py

# Full test suite
pytest testing/ -q

# With coverage
pytest testing/ --cov --cov-report=term-missing
```

---

Happy testing! 🚀
