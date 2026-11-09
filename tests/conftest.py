"""Fakes so tests need no network, no Ollama and no model downloads."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from memory_lab.llm import LLMReply


class FakeLLM:
    """Returns scripted replies in order and records every call."""

    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, json_mode=False, temperature=0.1, max_tokens=1500):
        self.calls.append(messages)
        r = self.replies.pop(0)
        text = r if isinstance(r, str) else json.dumps(r)
        return LLMReply(text, prompt_tokens=10, completion_tokens=5, seconds=0.0)


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get higher cosine similarity."""

    name = "fake-embedder"
    dims = 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


@pytest.fixture
def emb():
    return FakeEmbedder()
