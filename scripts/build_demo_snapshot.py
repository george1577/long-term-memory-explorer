"""Rebuild data/demo_store.json (the shipped demo memory store) from the 5 sample conversations.

    uv run python scripts/build_demo_snapshot.py [ollama-model]      # default qwen2.5:3b

Also writes data/demo_build_log.json: the facts and decisions for each conversation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_lab.demo import build_demo_store  # noqa: E402
from memory_lab.embedder import MiniLMEmbedder  # noqa: E402
from memory_lab.llm import OllamaLLM  # noqa: E402
from memory_lab.store import DEMO_SNAPSHOT, DEMO_USER  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
log = []


def on_step(i, conv, r):
    rec = r.reconciliation
    entry = {"conversation": i, "title": conv["title"], "facts": r.extraction.facts,
             "decisions": [{"event": d.event, "text": d.text, "old_text": d.old_text, "applied": d.applied,
                            "note": d.note} for d in rec.decisions],
             "shown_to_model": rec.old_memory}
    log.append(entry)
    print(json.dumps(entry, indent=1))


store, _ = build_demo_store(OllamaLLM(model), MiniLMEmbedder(), DEMO_USER, on_step)
store.save(DEMO_SNAPSHOT)
(DEMO_SNAPSHOT.parent / "demo_build_log.json").write_text(json.dumps(log, indent=1))
print(f"Saved {len(store.memories)} memories to {DEMO_SNAPSHOT}")
