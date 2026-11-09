"""Talking to a chat model: local Ollama models or OpenAI's gpt-4o-mini.

Every call returns an `LLMReply` with the text *and* the token counts and time taken, so the app can
show what each step of the memory pipeline costs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# label -> (provider, model name)
MODELS: dict[str, tuple[str, str]] = {
    "qwen2.5:3b (Ollama, local, free)": ("ollama", "qwen2.5:3b"),
    "qwen2.5:1.5b (Ollama, local, free, makes more mistakes)": ("ollama", "qwen2.5:1.5b"),
    "qwen2.5:7b (Ollama, local, free, slow on 8 GB)": ("ollama", "qwen2.5:7b"),
    "gpt-4o-mini (OpenAI, needs a key, what the book uses)": ("openai", "gpt-4o-mini"),
}
DEFAULT_MODEL = next(iter(MODELS))

# The context window we ask Ollama for. Its default is smaller than mem0's reconciliation prompt
# plus a long conversation, which would silently cut off the start of the prompt.
OLLAMA_NUM_CTX = 8192


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.1,
             max_tokens: int = 1500) -> LLMReply: ...


class OllamaLLM:
    def __init__(self, model: str, host: str = OLLAMA_URL):
        import ollama

        self.name = model
        self._client = ollama.Client(host=host)

    def chat(self, messages, *, json_mode=False, temperature=0.1, max_tokens=1500):
        t0 = time.perf_counter()
        r = self._client.chat(
            model=self.name,
            messages=messages,
            format="json" if json_mode else None,
            options={"temperature": temperature, "num_predict": max_tokens, "num_ctx": OLLAMA_NUM_CTX},
        )
        return LLMReply(
            text=r["message"]["content"],
            prompt_tokens=r.get("prompt_eval_count") or 0,
            completion_tokens=r.get("eval_count") or 0,
            seconds=time.perf_counter() - t0,
        )


class OpenAILLM:
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI

        self.name = model
        self._client = OpenAI(api_key=api_key)

    def chat(self, messages, *, json_mode=False, temperature=0.1, max_tokens=1500):
        t0 = time.perf_counter()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self._client.chat.completions.create(
            model=self.name, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return LLMReply(
            text=r.choices[0].message.content or "",
            prompt_tokens=r.usage.prompt_tokens if r.usage else 0,
            completion_tokens=r.usage.completion_tokens if r.usage else 0,
            seconds=time.perf_counter() - t0,
        )


def find_openai_key() -> str | None:
    """Look in the environment, then ./.env, then the book repo's examples/keys.env."""
    here = Path(__file__).resolve().parent.parent
    for f in (here / ".env", here.parent / "generative-ai-design-patterns" / "examples" / "keys.env"):
        if f.exists():
            load_dotenv(f, override=False)
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    # Template files often hold a placeholder like "# See https://..." or "your-openai-api-key".
    return key if key.startswith("sk-") else None


def ollama_models(host: str = OLLAMA_URL) -> list[str] | None:
    """Names of models pulled in Ollama, or None if Ollama isn't reachable."""
    try:
        import ollama

        return [m.model for m in ollama.Client(host=host, timeout=2).list().models]
    except Exception:
        return None


def make_llm(label: str, openai_key: str | None = None) -> LLM:
    provider, model = MODELS[label]
    if provider == "ollama":
        return OllamaLLM(model)
    if not openai_key:
        raise ValueError("gpt-4o-mini needs an OpenAI API key.")
    return OpenAILLM(model, openai_key)
