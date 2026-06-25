import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .config import settings


class VectorStore:
    def __init__(
        self,
        backend: str = settings.vector_backend,
        path: str = settings.vector_store_path,
    ):
        self.backend = backend
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings: List[List[float]] = []
        self.documents: List[dict] = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
                self.documents = payload.get("documents", [])
                self.embeddings = payload.get("embeddings", [])

    def _persist(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(
                {"documents": self.documents, "embeddings": self.embeddings},
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def add_documents(self, docs: List[dict]) -> None:
        for doc in docs:
            if any(existing["id"] == doc["id"] for existing in self.documents):
                continue
            self.documents.append(doc)
            self.embeddings.append(self.model.encode(doc["text"]).tolist())
        self._persist()

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.documents:
            return []

        query_embedding = self.model.encode(query).reshape(1, -1)
        matrix = np.array(self.embeddings)
        similarities = cosine_similarity(query_embedding, matrix).flatten()
        best = similarities.argsort()[::-1][:top_k]
        return [
            {
                "id": self.documents[index]["id"],
                "text": self.documents[index]["text"],
                "metadata": self.documents[index].get("metadata", {}),
                "score": float(similarities[index]),
            }
            for index in best
        ]


vector_store = VectorStore()
