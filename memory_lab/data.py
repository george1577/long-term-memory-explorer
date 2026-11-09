"""The notebook's sample conversations and queries, stored word for word in data/."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def conversations() -> list[dict]:
    """Each item: {"id", "title", "trap", "messages"}."""
    return json.loads((DATA / "sample_conversations.json").read_text())


@lru_cache(maxsize=1)
def queries() -> list[str]:
    return json.loads((DATA / "sample_queries.json").read_text())
