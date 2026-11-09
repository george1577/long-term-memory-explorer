from datetime import date

from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.utils import get_fact_retrieval_messages, parse_messages

from memory_lab import prompts


def test_extraction_prompt_matches_mem0_exactly():
    msgs = [{"role": "user", "content": "I love tea"}, {"role": "assistant", "content": "Nice!"}]
    sys_prompt, user = get_fact_retrieval_messages(parse_messages(msgs))
    assert prompts.fact_extraction_prompt(date.today()) == sys_prompt
    assert prompts.extraction_user_message(msgs) == user


def test_reconciliation_prompt_matches_mem0_exactly():
    old = [{"id": "0", "text": "Likes tea"}]
    assert prompts.update_memory_message(old, ["Loves green tea"]) == get_update_memory_messages(old, ["Loves green tea"])


def test_answer_prompt_without_memories_is_plain():
    assert prompts.answer_system_prompt([]) == "You are a helpful AI assistant."
    assert "- Likes tea" in prompts.answer_system_prompt(["Likes tea"])
