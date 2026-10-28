from pathlib import Path
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import get_research_history, init_db, save_research
from research_assistant.graph import research_graph

app = FastAPI(
    title="Beginner-Friendly Autonomous Research Assistant",
    description="A clean, understandable AI research assistant built with LangGraph, FastAPI, and SQLite.",
    version="1.0.0",
)

# Enable CORS for easy local exploration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if present
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def on_startup() -> None:
    """Initialize the SQLite database on application startup."""
    init_db()


# ---------------------------------------------------------------------------
# Request and Response Models (Pydantic)
# ---------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The research question to investigate.",
        example="What are the practical applications of AI agents?",
    )


class ResearchResponse(BaseModel):
    question: str
    search_queries: List[str]
    final_report: str


class HistoryItem(BaseModel):
    id: int
    question: str
    search_queries: List[str]
    final_report: str
    created_at: str


class HistoryResponse(BaseModel):
    history: List[HistoryItem]


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse, include_in_schema=False)
def home():
    """Serve the static frontend if available, or return health."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "ok", "message": "Autonomous Research Assistant API is running."})


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint to verify that the service is running."""
    return {"status": "ok"}


@app.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Research"],
)
def run_research(request: ResearchRequest):
    """Accepts a research question, executes the 3-node LangGraph workflow,

    saves the result to SQLite, and returns the generated report and queries.
    """
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research question cannot be empty or blank.",
        )

    try:
        # 1. Execute the LangGraph workflow
        # graph.invoke() starts execution at START and flows through all nodes to END
        result = research_graph.invoke({"question": clean_question})

        # 2. Extract outputs from the final graph state
        final_report = result.get("final_report", "")
        search_queries = result.get("search_queries", [])

        # 3. Persist successful research execution to SQLite
        save_research(
            question=clean_question,
            search_queries=search_queries,
            final_report=final_report,
        )

        return ResearchResponse(
            question=clean_question,
            search_queries=search_queries,
            final_report=final_report,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while running the research workflow: {str(e)}",
        )


@app.get(
    "/research/history",
    response_model=HistoryResponse,
    tags=["Research"],
)
def list_history(
    limit: int = Query(default=10, ge=1, le=100, description="Max number of recent records to return"),
):
    """Returns the most recent research reports saved in SQLite."""
    records = get_research_history(limit=limit)
    return HistoryResponse(history=[HistoryItem(**rec) for rec in records])


# ---------------------------------------------------------------------------
# Backward Compatibility Endpoints for existing frontend
# ---------------------------------------------------------------------------
@app.get("/api/health", include_in_schema=False)
def api_health():
    return {"status": "ok"}


@app.post("/api/research", include_in_schema=False)
def api_research(request: ResearchRequest):
    result = run_research(request)
    return {
        "question": result.question,
        "plan": result.search_queries,
        "results": [{"source": "web", "summary": q, "details": ""} for q in result.search_queries],
        "final_report": result.final_report,
    }
