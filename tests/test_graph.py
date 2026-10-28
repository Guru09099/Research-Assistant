"""Unit tests for LangGraph nodes and the compiled workflow."""

from unittest.mock import MagicMock, patch
import pytest

from research_assistant.graph import create_research_graph, research_graph
from research_assistant.nodes import generate_report, plan_queries, search_web
from research_assistant.state import ResearchState


def test_plan_queries_fallback():
    """Test that plan_queries produces 2-3 queries even without an LLM key."""
    state: ResearchState = {
        "question": "What are the practical applications of AI agents?",
        "search_queries": [],
        "search_results": [],
        "final_report": "",
    }
    output = plan_queries(state)
    queries = output.get("search_queries", [])
    
    assert isinstance(queries, list)
    assert 2 <= len(queries) <= 3
    assert any("AI agents" in q for q in queries)


def test_plan_queries_with_mocked_llm():
    """Test that plan_queries properly parses structured LLM output."""
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [
        MagicMock(message=MagicMock(content='["AI agents in healthcare", "AI agents in customer service"]'))
    ]

    with patch("config.OPENAI_API_KEY", "mock-key"):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_llm_response
            mock_openai.return_value = mock_client

            state: ResearchState = {
                "question": "AI agent applications",
                "search_queries": [],
                "search_results": [],
                "final_report": "",
            }
            output = plan_queries(state)
            assert output["search_queries"] == [
                "AI agents in healthcare",
                "AI agents in customer service",
            ]


def test_search_web_deduplicates_urls():
    """Test that search_web removes duplicate URLs across different queries."""
    mock_tavily_results = {
        "results": [
            {"title": "Doc 1", "url": "https://example.com/agents", "content": "Text 1"},
            {"title": "Doc 2", "url": "https://example.com/agents", "content": "Duplicate URL"},
            {"title": "Doc 3", "url": "https://example.com/unique", "content": "Unique Text"},
        ]
    }

    with patch("config.TAVILY_API_KEY", "mock-tavily-key"):
        with patch("tavily.TavilyClient") as mock_tavily_client_cls:
            mock_client = MagicMock()
            mock_client.search.return_value = mock_tavily_results
            mock_tavily_client_cls.return_value = mock_client

            state: ResearchState = {
                "question": "AI agents",
                "search_queries": ["query 1"],
                "search_results": [],
                "final_report": "",
            }
            output = search_web(state)
            results = output.get("search_results", [])

            # Only 2 unique URLs should remain out of 3
            assert len(results) == 2
            urls = [r["url"] for r in results]
            assert urls == ["https://example.com/agents", "https://example.com/unique"]


def test_search_web_handles_empty_queries():
    """Test that search_web safely handles empty search queries list."""
    state: ResearchState = {
        "question": "AI agents",
        "search_queries": [],
        "search_results": [],
        "final_report": "",
    }
    output = search_web(state)
    assert output["search_results"] == []


def test_generate_report_handles_empty_results():
    """Test that generate_report produces a helpful message when no evidence is found."""
    state: ResearchState = {
        "question": "Quantum computing advances",
        "search_queries": ["quantum computing"],
        "search_results": [],
        "final_report": "",
    }
    output = generate_report(state)
    report = output.get("final_report", "")

    assert "could not be completed reliably" in report
    assert "Quantum computing advances" in report


def test_generate_report_with_evidence():
    """Test that generate_report synthesizes evidence and includes source links."""
    state: ResearchState = {
        "question": "Applications of AI agents",
        "search_queries": ["AI agents"],
        "search_results": [
            {
                "title": "Agentic Workflows in Production",
                "url": "https://example.com/agents-guide",
                "content": "AI agents can automate complex multistep business processes.",
            }
        ],
        "final_report": "",
    }
    output = generate_report(state)
    report = output.get("final_report", "")

    assert "Applications of AI agents" in report
    assert "https://example.com/agents-guide" in report
    assert "Agentic Workflows in Production" in report
    assert "Summary" in report
    assert "Key Findings" in report


def test_graph_end_to_end_execution():
    """Test that the full 3-node graph executes from START to END."""
    graph = create_research_graph()
    initial_state = {"question": "What is reinforcement learning?"}
    
    final_state = graph.invoke(initial_state)

    # Verify that all 4 state fields are populated upon reaching END
    assert final_state["question"] == "What is reinforcement learning?"
    assert isinstance(final_state["search_queries"], list)
    assert len(final_state["search_queries"]) >= 2
    assert isinstance(final_state["search_results"], list)
    assert len(final_state["search_results"]) >= 1
    assert "What is reinforcement learning?" in final_state["final_report"]

