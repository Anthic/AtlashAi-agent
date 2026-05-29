"""
conftest.py
────────────────────────────────────────────────────────────
Pytest configuration and shared fixtures for all tests.
Place this in the testing/ directory.

Automatically runs before each test to set up environment.
"""

import sys
import os
from pathlib import Path

import pytest
from unittest.mock import MagicMock


# ──────────────────────────────────────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────────────────────────────────────

# Add parent directory to path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create outputs directory for test artifacts
outputs_dir = project_root / "outputs"
outputs_dir.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load Environment
# ──────────────────────────────────────────────────────────────────────────────

# Load .env if it exists
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)


# ──────────────────────────────────────────────────────────────────────────────
# Pytest Hooks
# ──────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure pytest before test session"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that need real API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Modify tests during collection"""
    for item in items:
        # Add markers based on test name or file
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# ──────────────────────────────────────────────────────────────────────────────
# Global Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_config():
    """Shared test configuration"""
    return {
        "max_retries": 2,
        "model": "mistral-large",
        "timeout": 120,
        "test_topic": "machine learning ethics",
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for tests that don't call real API"""
    llm = MagicMock()
    llm.invoke.return_value = "Mock LLM response"
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model"""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return embeddings


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for RAG tests"""
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [
        MagicMock(page_content="Mock document 1"),
        MagicMock(page_content="Mock document 2"),
    ]
    return vectorstore


@pytest.fixture
def sample_state():
    """Sample research state for testing"""
    return {
        "topic": "quantum computing applications",
        "plan": "Research plan",
        "sub_questions": ["Q1", "Q2", "Q3"],
        "search_queries": ["query1", "query2"],
        "rewritten_queries": [],
        "search_topic": "quantum computing",
        "search_results": ["result1", "result2"],
        "verified_urls": [
            "https://example.com/page1",
            "https://example.com/page2",
        ],
        "urls": [
            "https://example.com/page1",
            "https://example.com/page2",
        ],
        "scraped_content": "Sample content from URLs",
        "summarized_content": "Summary of content",
        "rag_context": "Context from vector store",
        "report": "Generated research report",
        "draft_sections": ["Section 1", "Section 2"],
        "knowledge_graph": {
            "nodes": [{"id": "1", "label": "Node 1"}],
            "edges": [{"source": "1", "target": "2"}],
        },
        "knowledge_graph_md": "Knowledge graph markdown",
        "critique": "Critique feedback",
        "critique_score": 8,
        "retry_count": 0,
        "max_retries": 2,
        "fact_check_result": "Fact check results",
        "fact_check_score": 0.9,
        "error": "",
        "time_sec": 1.5,
    }


@pytest.fixture
def mock_search_results():
    """Mock search results"""
    return {
        "search_results": [
            "Result 1: Information about the topic",
            "Result 2: More relevant information",
        ],
        "verified_urls": [
            "https://source1.com/article",
            "https://source2.org/research",
        ],
        "urls": [
            "https://source1.com/article",
            "https://source2.org/research",
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pytest Plugins & Extensions
# ──────────────────────────────────────────────────────────────────────────────

# Note: pytest_runtest_setup removed - nodeid is read-only in pytest 9.0+
# Docstrings are shown in verbose output automatically


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures for Async Tests (if using pytest-asyncio)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def async_mock_llm():
    """Async mock LLM"""
    llm = MagicMock()
    llm.ainvoke = MagicMock(return_value="Async mock response")
    return llm


# ──────────────────────────────────────────────────────────────────────────────
# Test Report Customization
# ──────────────────────────────────────────────────────────────────────────────

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary at end of test run"""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "✓ ALL TESTS PASSED", green=True, bold=True)
    else:
        terminalreporter.write_sep("=", "✗ SOME TESTS FAILED", red=True, bold=True)


# ──────────────────────────────────────────────────────────────────────────────
# Parametrized Test Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[
    "machine learning ethics",
    "quantum computing",
    "climate change",
    "blockchain technology",
])
def various_topics(request):
    """Fixture providing multiple topics for parametrized tests"""
    return request.param


@pytest.fixture(params=[
    {"score": 9, "expected": "pass"},
    {"score": 8, "expected": "pass"},
    {"score": 7, "expected": "retry"},
    {"score": 6, "expected": "retry"},
])
def critique_scores(request):
    """Fixture providing various critique scores"""
    return request.param
