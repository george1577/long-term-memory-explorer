"""Building the shared demo memory store from the notebook's 5 sample conversations."""

from __future__ import annotations

from typing import Callable

from . import data
from .embedder import Embedder
from .llm import LLM
from .pipeline import AddResult, add_conversation
from .store import MemoryStore


def build_demo_store(llm: LLM, embedder: Embedder, user_id: str,
                     on_step: Callable[[int, dict, AddResult], None] | None = None
                     ) -> tuple[MemoryStore, list[AddResult]]:
    """Start from an empty store and add the 5 conversations one after another, like the notebook."""
    store = MemoryStore(user_id, built_with=llm.name)
    results = []
    for conv in data.conversations():
        res = add_conversation(llm, embedder, store, conv["messages"], source=f"Conversation {conv['id']}")
        results.append(res)
        if on_step:
            on_step(conv["id"], conv, res)
    return store, results
