"""Integration tests for FastAPI endpoints."""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from app.main import app
import database


@pytest.fixture
def client(tmp_path):
    """Provides a TestClient connected to an isolated test SQLite database."""
    test_db = str(tmp_path / "test_research.db")
    database.init_db(test_db)

    with patch("database.DATABASE_PATH", test_db):
        with patch("config.DATABASE_PATH", test_db):
            with TestClient(app) as test_client:
                yield test_client


def test_health_endpoint(client):
    """GET /health must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_rejects_empty_question(client):
    """POST /research must reject empty or whitespace questions."""
    # Empty string validation via FastAPI Pydantic / HTTPException
    response = client.post("/research", json={"question": ""})
    assert response.status_code in (400, 422)

    response_spaces = client.post("/research", json={"question": "   "})
    assert response_spaces.status_code in (400, 422)


def test_research_endpoint_success(client):
    """POST /research executes the graph, saves to DB, and returns structured response."""
    payload = {"question": "What are the practical applications of AI agents?"}
    response = client.post("/research", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "What are the practical applications of AI agents?"
    assert isinstance(data["search_queries"], list)
    assert len(data["search_queries"]) >= 2
    assert "final_report" in data
    assert len(data["final_report"]) > 0


def test_research_history_endpoint(client):
    """GET /research/history returns records saved during research execution."""
    # Run a research query first
    client.post("/research", json={"question": "History test question"})

    # Check history
    response = client.get("/research/history?limit=5")
    assert response.status_code == 200
    data = response.json()

    assert "history" in data
    assert len(data["history"]) >= 1

    latest = data["history"][0]
    assert latest["question"] == "History test question"
    assert isinstance(latest["search_queries"], list)
    assert "final_report" in latest
    assert "created_at" in latest

