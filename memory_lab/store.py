"""The memory store: every memory kept for one user, saved as a plain JSON file.

A real system would use a vector database (the book's example uses Chroma). A JSON file does the same
job for a few hundred memories, and you can open it in any editor to see exactly what was remembered.
Every change is also written to a history log, like mem0's history database.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STORES_DIR = ROOT / "data" / "stores"
DEMO_SNAPSHOT = ROOT / "data" / "demo_store.json"
DEMO_USER = "demo_user"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MemoryItem:
    id: str
    text: str
    embedding: list[float]
    created_at: str
    updated_at: str


@dataclass
class HistoryEntry:
    time: str
    memory_id: str
    event: str  # ADD, UPDATE or DELETE
    old_text: str | None
    new_text: str | None
    source: str = ""  # what caused the change, e.g. "Conversation 3" or "Tab 3"


@dataclass
class Hit:
    item: MemoryItem
    score: float  # cosine similarity: 1 = same meaning, 0 = unrelated


@dataclass
class MemoryStore:
    user_id: str
    memories: list[MemoryItem] = field(default_factory=list)
    history: list[HistoryEntry] = field(default_factory=list)
    built_with: str = ""  # which LLM produced these memories

    # ---- reading -------------------------------------------------------------------------------
    def get(self, memory_id: str) -> MemoryItem | None:
        return next((m for m in self.memories if m.id == memory_id), None)

    def search(self, query_vec: np.ndarray, limit: int | None = None,
               threshold: float | None = None) -> list[Hit]:
        """Every memory ranked by cosine similarity to the query, best first."""
        if not self.memories:
            return []
        matrix = np.asarray([m.embedding for m in self.memories], dtype=np.float32)
        scores = matrix @ np.asarray(query_vec, dtype=np.float32)
        hits = [Hit(m, float(s)) for m, s in zip(self.memories, scores)]
        hits.sort(key=lambda h: h.score, reverse=True)
        if threshold is not None:
            hits = [h for h in hits if h.score >= threshold]
        return hits[:limit] if limit is not None else hits

    # ---- writing -------------------------------------------------------------------------------
    def add(self, text: str, embedding: np.ndarray, source: str = "") -> MemoryItem:
        now = _now()
        item = MemoryItem(str(uuid.uuid4()), text, _vec(embedding), now, now)
        self.memories.append(item)
        self.history.append(HistoryEntry(now, item.id, "ADD", None, text, source))
        return item

    def update(self, memory_id: str, text: str, embedding: np.ndarray, source: str = "") -> MemoryItem:
        item = self._require(memory_id)
        old = item.text
        item.text, item.embedding, item.updated_at = text, _vec(embedding), _now()
        self.history.append(HistoryEntry(item.updated_at, memory_id, "UPDATE", old, text, source))
        return item

    def delete(self, memory_id: str, source: str = "") -> MemoryItem:
        item = self._require(memory_id)
        self.memories.remove(item)
        self.history.append(HistoryEntry(_now(), memory_id, "DELETE", item.text, None, source))
        return item

    def clear(self) -> None:
        self.memories.clear()
        self.history.clear()

    def _require(self, memory_id: str) -> MemoryItem:
        item = self.get(memory_id)
        if item is None:
            raise KeyError(memory_id)
        return item

    # ---- files ---------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "built_with": self.built_with,
            "memories": [asdict(m) for m in self.memories],
            "history": [asdict(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict, user_id: str | None = None) -> MemoryStore:
        return cls(
            user_id=user_id or d["user_id"],
            memories=[MemoryItem(**m) for m in d.get("memories", [])],
            history=[HistoryEntry(**h) for h in d.get("history", [])],
            built_with=d.get("built_with", ""),
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or store_path(self.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path


def _vec(v: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in np.asarray(v).ravel()]


def store_path(user_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id) or "user"
    return STORES_DIR / f"{safe}.json"


def load_store(user_id: str) -> MemoryStore:
    """Load a user's store. The demo user starts from the shipped snapshot; anyone else starts empty."""
    path = store_path(user_id)
    if path.exists():
        return MemoryStore.from_dict(json.loads(path.read_text()), user_id)
    if user_id == DEMO_USER and DEMO_SNAPSHOT.exists():
        return MemoryStore.from_dict(json.loads(DEMO_SNAPSHOT.read_text()), user_id)
    return MemoryStore(user_id)


def reset_to_snapshot(user_id: str) -> MemoryStore:
    """Replace a user's store with the shipped demo snapshot."""
    store = MemoryStore.from_dict(json.loads(DEMO_SNAPSHOT.read_text()), user_id)
    store.save()
    return store
