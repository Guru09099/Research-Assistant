# Autonomous Research Assistant

A FastAPI-based research assistant that can decompose research questions, invoke tool-based searches, analyze PDFs, store memory, and generate structured reports.

## Features

- Query decomposition into research steps
- Web search via SerpAPI
- PDF ingestion and similarity search
- Persistent memory storage
- FastAPI endpoints for research and uploads
- Modular vector search with sentence embeddings
- GitHub Actions workflow included

## Getting Started

1. Clone the repository
2. Create a Python environment
3. Install dependencies

```powershell
python -m venv venv
```

If PowerShell script execution is blocked, use one of these two options:

```powershell
# Option 1: bypass activation for current session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

```powershell
# Option 2: use the interpreter directly without activation
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Create `.env` based on `.env.example`
5. Start the app

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

- `OPENAI_API_KEY` - required for LLM planning and report generation, or alternatively use `GEMINI_API_KEY`
- `GEMINI_API_KEY` - optional Gemini-compatible API key for LLM access
- `SERPAPI_API_KEY` - optional for live web search
- `VECTOR_BACKEND` - optional vector search backend name
- `PDF_STORAGE_DIR` - optional local PDF storage directory

## Endpoints

- `GET /` - research assistant frontend UI
- `GET /api/health`
- `POST /api/research` - run research on a question
- `POST /api/upload-pdf` - upload PDF and ingest into vector store
- `GET /api/memory` - list memory entries
- `POST /api/memory` - add a memory item

## Frontend

The project now includes a lightweight web frontend served from `app/static`. Open the app in the browser at `http://localhost:8000` after startup.

## Architecture

- `app/main.py` - FastAPI interface
- `app/agent.py` - planning and tool orchestration
- `app/tools.py` - web search, PDF parsing, memory search
- `app/vector_store.py` - embedding search store
- `app/memory.py` - JSON-backed memory persistence

## Deploying

Use the provided GitHub Actions workflow or containerize with Docker.
