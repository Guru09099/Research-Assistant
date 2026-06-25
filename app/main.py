from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import ResearchAgent
from .config import settings
from .memory import memory_store
from .schemas import AgentResponse, MemoryItem, QueryRequest, UploadResponse
from .tools import analyze_pdf, ingest_pdf_to_vector_store

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

agent = ResearchAgent(use_langchain=settings.use_langchain)


@app.on_event("startup")
def startup_event() -> None:
    Path(settings.pdf_storage_dir).mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "application": settings.app_name})


@app.post("/api/research", response_model=AgentResponse)
def research(request: QueryRequest) -> AgentResponse:
    result = agent.run(
        query=request.question,
        use_web_search=request.use_web_search,
        use_pdf_search=request.use_pdf_search,
        use_memory=request.use_memory,
    )
    return AgentResponse(**result)


@app.post("/api/upload-pdf", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    filename = Path(file.filename).name
    safe_path = Path(settings.pdf_storage_dir) / filename
    with open(safe_path, "wb") as buffer:
        buffer.write(file.file.read())
    pages = analyze_pdf(str(safe_path))
    ingest_pdf_to_vector_store(str(safe_path), file_name=filename)
    return UploadResponse(filename=filename, pages=pages, status="uploaded")


@app.get("/api/memory", response_model=List[MemoryItem])
def read_memory() -> List[MemoryItem]:
    return memory_store.list()


@app.post("/api/memory", response_model=MemoryItem)
def add_memory(item: MemoryItem) -> MemoryItem:
    memory_store.add(item)
    return item
