import json
from pathlib import Path
from typing import List

from .config import settings
from .schemas import MemoryItem


class MemoryStore:
    def __init__(self, path: str = settings.memory_store_path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return []

    def _persist(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._store, handle, indent=2, ensure_ascii=False)

    def add(self, item: MemoryItem) -> None:
        self._store.append(item.dict())
        self._persist()

    def list(self) -> List[MemoryItem]:
        return [MemoryItem(**item) for item in self._store]

    def search(self, query: str) -> List[MemoryItem]:
        lower_query = query.lower()
        results = []
        for item in self._store:
            tags = item.get("tags", [])
            if (
                lower_query in item["content"].lower()
                or any(lower_query in tag.lower() for tag in tags)
            ):
                results.append(MemoryItem(**item))
        return results


memory_store = MemoryStore()
