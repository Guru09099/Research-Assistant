from typing import List
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    use_web_search: bool = True
    use_pdf_search: bool = True
    use_memory: bool = True


class MemoryItem(BaseModel):
    id: str
    content: str
    tags: List[str] = []


class ToolResult(BaseModel):
    source: str
    summary: str
    details: str


class AgentResponse(BaseModel):
    question: str
    plan: List[str]
    results: List[ToolResult]
    final_report: str


class UploadResponse(BaseModel):
    filename: str
    pages: int
    status: str
