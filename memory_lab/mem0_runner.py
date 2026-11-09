"""Running the real mem0 library (version 0.1.113, as pinned by the book) for comparison.

mem0 gets the same LLM and the same embedding model as the rest of the app, and a fresh, temporary
Chroma database every run, just like the book's `get_memory_config()`.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

# mem0 sends usage telemetry by default and writes to ~/.mem0. Turn both off before importing it.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("MEM0_DIR", os.path.join(tempfile.gettempdir(), "mem0_explorer_home"))

from .embedder import EMBEDDING_MODEL  # noqa: E402
from .llm import MODELS, OLLAMA_URL  # noqa: E402


@dataclass
class Mem0Search:
    query: str
    results: list[dict]  # each: {"memory": str, "distance": float, "cosine": float}


@dataclass
class Mem0Run:
    events: list[list[dict]] = field(default_factory=list)  # per conversation: what mem0.add returned
    memories: list[str] = field(default_factory=list)       # mem0.get_all at the end
    searches: list[Mem0Search] = field(default_factory=list)
    seconds: float = 0.0
    storage_dir: str = ""


def mem0_config(model_label: str, openai_key: str | None, storage_dir: str) -> dict:
    provider, model = MODELS[model_label]
    llm_cfg = {"model": model, "temperature": 0.1, "max_tokens": 1500}
    if provider == "ollama":
        llm_cfg["ollama_base_url"] = OLLAMA_URL
    else:
        llm_cfg["api_key"] = openai_key
    return {
        "vector_store": {"provider": "chroma", "config": {
            "collection_name": "mem0_explorer", "path": os.path.join(storage_dir, "chroma_db")}},
        "llm": {"provider": provider, "config": llm_cfg},
        # Force CPU: sentence-transformers would pick Apple's MPS, which crashes PyTorch on macOS 14.
        "embedder": {"provider": "huggingface", "config": {"model": EMBEDDING_MODEL,
                                                           "model_kwargs": {"device": "cpu"}}},
        "history_db_path": os.path.join(storage_dir, "history.db"),
        "version": "v1.1",
    }


def run_mem0(conversations: list[dict], queries: list[str], model_label: str, openai_key: str | None,
             limit: int = 2, user_id: str = "demo_user",
             on_step: Callable[[str], None] | None = None) -> Mem0Run:
    from mem0 import Memory

    storage_dir = tempfile.mkdtemp(prefix="mem0_explorer_")
    run = Mem0Run(storage_dir=storage_dir)
    t0 = time.perf_counter()
    memory = Memory.from_config(mem0_config(model_label, openai_key, storage_dir))
    for conv in conversations:
        if on_step:
            on_step(f"mem0.add: conversation {conv['id']} ({conv['title']})")
        out = memory.add(conv["messages"], user_id=user_id)
        run.events.append(out.get("results", []) if isinstance(out, dict) else list(out))
    all_mem = memory.get_all(user_id=user_id)
    run.memories = [m["memory"] for m in (all_mem.get("results", []) if isinstance(all_mem, dict) else all_mem)]
    for q in queries:
        if on_step:
            on_step(f"mem0.search: {q}")
        out = memory.search(query=q, user_id=user_id, limit=limit)
        rows = out.get("results", []) if isinstance(out, dict) else out
        run.searches.append(Mem0Search(q, [_row(r) for r in rows]))
    run.seconds = time.perf_counter() - t0
    return run


def _row(r: dict) -> dict:
    # With Chroma, mem0's "score" is a squared L2 *distance* (lower = closer), not a similarity.
    # MiniLM vectors have length 1, so cosine similarity = 1 - distance / 2.
    d = r.get("score")
    return {"memory": r.get("memory", ""), "distance": d,
            "cosine": None if d is None else 1 - d / 2}
