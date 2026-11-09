"""Counting tokens so the app can compare the cost of different ways of giving the model context."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")  # gpt-4o's tokenizer; close enough for comparisons


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def count_message_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) + 4 for m in messages)
