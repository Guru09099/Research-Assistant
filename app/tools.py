import re
from pathlib import Path
from typing import List

import requests
from PyPDF2 import PdfReader

from .config import settings
from .vector_store import vector_store


def query_web(query: str, top_k: int = 5) -> dict:
    if not settings.serpapi_api_key:
        return {
            "source": "web_search",
            "summary": "Web search skipped: SERPAPI_API_KEY is not configured.",
            "details": "Set SERPAPI_API_KEY in your .env file to enable live search.",
        }

    response = requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,
            "engine": "google",
            "api_key": settings.serpapi_api_key,
            "num": top_k,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    snippets = []
    for item in payload.get("organic_results", [])[:top_k]:
        title = item.get("title") or "No title"
        snippet = item.get("snippet", "No snippet available.")
        link = item.get("link") or item.get("snippet")
        snippets.append(f"{title}: {snippet} ({link})")

    return {
        "source": "web_search",
        "summary": f"Collected {len(snippets)} web search results for query: {query}",
        "details": "\n".join(snippets),
    }


def analyze_pdf(file_path: str) -> int:
    reader = PdfReader(file_path)
    return len(reader.pages)


def extract_pdf_text(file_path: str) -> List[dict]:
    reader = PdfReader(file_path)
    documents = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            continue
        documents.append(
            {
                "id": f"pdf:{Path(file_path).stem}:{idx}",
                "text": cleaned,
                "metadata": {"source": Path(file_path).name, "page": idx},
            }
        )
    return documents


def ingest_pdf_to_vector_store(file_path: str, file_name: str | None = None) -> None:
    docs = extract_pdf_text(file_path)
    if not docs:
        return
    vector_store.add_documents(docs)


def search_pdf(query: str, top_k: int = 5) -> dict:
    matches = vector_store.search(query, top_k=top_k)
    if not matches:
        return {
            "source": "pdf_search",
            "summary": "No relevant PDF content found.",
            "details": "Upload documents and retry the search.",
        }
    formatted = []
    for item in matches:
        formatted.append(
            "Page {page} ({source}): {text}".format(
                page=item["metadata"].get("page"),
                source=item["metadata"].get("source"),
                text=item["text"][:280],
            )
        )
    return {
        "source": "pdf_search",
        "summary": f"Found {len(matches)} relevant PDF passages.",
        "details": "\n".join(formatted),
    }


def search_memory(query: str) -> dict:
    from .memory import memory_store

    memories = memory_store.search(query)
    if not memories:
        return {
            "source": "memory",
            "summary": "No matching memories found.",
            "details": "Try storing more project knowledge or user preferences first.",
        }
    details = "\n".join([f"{item.id}: {item.content}" for item in memories])
    return {
        "source": "memory",
        "summary": f"Retrieved {len(memories)} memory items matching the query.",
        "details": details,
    }
