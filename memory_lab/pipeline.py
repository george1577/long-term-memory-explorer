"""The memory pipeline, step by step, with every intermediate result kept so the app can show it.

This follows mem0 0.1.113's `Memory.add` and `Memory.search` (the library the book's example uses):

  write path:  conversation --Extraction--> candidate facts --Reconciliation--> ADD / UPDATE / DELETE / NONE
  read path:   query --Retrieval--> top memories --> memory-augmented prompt --> answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import prompts
from .embedder import Embedder
from .llm import LLM, LLMReply
from .parsing import parse_json_object
from .store import Hit, MemoryStore

# mem0 compares each candidate fact with this many similar existing memories.
CANDIDATES_PER_FACT = 5
EVENTS = ("ADD", "UPDATE", "DELETE", "NONE")


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    system_prompt: str
    user_message: str
    reply: LLMReply
    facts: list[str]
    parsed: bool  # False if the model's reply wasn't valid JSON (then facts is empty, like mem0)


def extract(llm: LLM, messages: list[dict], today: date | None = None) -> ExtractionResult:
    system_prompt = prompts.fact_extraction_prompt(today)
    user_message = prompts.extraction_user_message(messages)
    reply = llm.chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
        json_mode=True,
    )
    obj = parse_json_object(reply.text)
    raw = obj.get("facts") if obj else None
    facts = [str(f).strip() for f in raw if str(f).strip()] if isinstance(raw, list) else []
    return ExtractionResult(system_prompt, user_message, reply, facts, parsed=isinstance(raw, list))


# ---------------------------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    """An existing memory found similar to one candidate fact."""
    fact: str
    memory_id: str
    text: str
    score: float


@dataclass
class Decision:
    event: str           # ADD, UPDATE, DELETE, NONE (or whatever the model wrote)
    llm_id: str          # the short id the model used ("0", "1", ...)
    memory_id: str | None  # the real id it maps to (None for ADD before saving, or if invalid)
    text: str
    old_text: str | None
    applied: bool
    note: str = ""


@dataclass
class ReconciliationResult:
    facts: list[str]
    candidates: list[Candidate]           # per fact, before de-duplication
    id_map: dict[str, str]                # short id shown to the model -> real memory id
    old_memory: list[dict]                # what the model was shown as "current memory"
    prompt: str | None                    # None when there were no facts (the LLM isn't called)
    reply: LLMReply | None
    parsed: bool
    decisions: list[Decision] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = {e: 0 for e in EVENTS}
        for d in self.decisions:
            if d.applied or d.event == "NONE":
                c[d.event] = c.get(d.event, 0) + 1
        return c


def reconcile(llm: LLM, embedder: Embedder, store: MemoryStore, facts: list[str],
              source: str = "") -> ReconciliationResult:
    if not facts:
        return ReconciliationResult(facts, [], {}, [], None, None, True)

    # 1. For every candidate fact, find the most similar existing memories.
    fact_vecs = embedder.embed(facts)
    vec_by_text = {f: v for f, v in zip(facts, fact_vecs)}
    candidates: list[Candidate] = []
    for fact, vec in zip(facts, fact_vecs):
        for hit in store.search(vec, limit=CANDIDATES_PER_FACT):
            candidates.append(Candidate(fact, hit.item.id, hit.item.text, hit.score))

    # 2. De-duplicate, then swap long ids for "0", "1", ... (mem0 does this because models garble UUIDs).
    unique: dict[str, str] = {}
    for c in candidates:
        unique.setdefault(c.memory_id, c.text)
    id_map = {str(i): mid for i, mid in enumerate(unique)}
    old_memory = [{"id": short, "text": unique[mid]} for short, mid in id_map.items()]

    # 3. Ask the model to decide.
    prompt = prompts.update_memory_message(old_memory, facts)
    reply = llm.chat([{"role": "user", "content": prompt}], json_mode=True)
    obj = parse_json_object(reply.text)
    entries = obj.get("memory") if obj else None
    result = ReconciliationResult(facts, candidates, id_map, old_memory, prompt, reply,
                                  parsed=isinstance(entries, list))

    # 4. Apply the decisions to the store.
    for e in entries if isinstance(entries, list) else []:
        if not isinstance(e, dict):
            continue
        result.decisions.append(_apply(e, store, embedder, id_map, vec_by_text, source))
    return result


def _apply(e: dict, store: MemoryStore, embedder: Embedder, id_map: dict[str, str],
           vec_by_text: dict[str, np.ndarray], source: str) -> Decision:
    event = str(e.get("event", "")).upper()
    llm_id = str(e.get("id", ""))
    text = str(e.get("text") or "").strip()
    old_text = e.get("old_memory")
    real_id = id_map.get(llm_id)

    def vec_for(t: str) -> np.ndarray:
        return vec_by_text[t] if t in vec_by_text else embedder.embed([t])[0]

    if not text:
        return Decision(event, llm_id, real_id, text, old_text, False, "skipped: empty text")
    if event == "ADD":
        item = store.add(text, vec_for(text), source)
        return Decision(event, llm_id, item.id, text, None, True)
    if event in ("UPDATE", "DELETE"):
        if real_id is None or store.get(real_id) is None:
            return Decision(event, llm_id, None, text, old_text, False,
                            f"skipped: id {llm_id!r} was not one of the memories shown to the model")
        if event == "UPDATE":
            before = store.get(real_id).text
            if before == text:
                return Decision(event, llm_id, real_id, text, before, False, "skipped: text unchanged")
            store.update(real_id, text, vec_for(text), source)
            return Decision(event, llm_id, real_id, text, before, True)
        before = store.delete(real_id, source).text
        return Decision(event, llm_id, real_id, before, None, True)
    if event == "NONE":
        return Decision(event, llm_id, real_id, text, None, False)
    return Decision(event, llm_id, real_id, text, old_text, False, f"skipped: unknown event {event!r}")


# ---------------------------------------------------------------------------------------------
# Write path = Extraction + Reconciliation
# ---------------------------------------------------------------------------------------------
@dataclass
class AddResult:
    extraction: ExtractionResult
    reconciliation: ReconciliationResult


def add_conversation(llm: LLM, embedder: Embedder, store: MemoryStore, messages: list[dict],
                     source: str = "", today: date | None = None) -> AddResult:
    ex = extract(llm, messages, today)
    rec = reconcile(llm, embedder, store, ex.facts, source)
    return AddResult(ex, rec)


# ---------------------------------------------------------------------------------------------
# Read path = Retrieval + memory-augmented answer
# ---------------------------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    query: str
    ranked: list[Hit]   # every memory, best first
    selected: list[Hit]  # what passes the limit and threshold


def retrieve(embedder: Embedder, store: MemoryStore, query: str, limit: int = 3,
             threshold: float | None = None) -> RetrievalResult:
    ranked = store.search(embedder.embed([query])[0])
    selected = [h for h in ranked if threshold is None or h.score >= threshold][:limit]
    return RetrievalResult(query, ranked, selected)


@dataclass
class AnswerResult:
    messages: list[dict]
    reply: LLMReply


def answer(llm: LLM, query: str, system_prompt: str, history: list[dict] | None = None) -> AnswerResult:
    """Answer a query with the given system prompt (and, for chat, the conversation history so far)."""
    messages = [{"role": "system", "content": system_prompt}, *(history or []),
                {"role": "user", "content": query}]
    # Same settings as the book's example: temperature 0.7, up to 500 tokens.
    return AnswerResult(messages, llm.chat(messages, temperature=0.7, max_tokens=500))


@dataclass
class TurnResult:
    retrieval: RetrievalResult
    answer: AnswerResult
    write: AddResult | None


def memory_turn(llm: LLM, embedder: Embedder, store: MemoryStore, query: str, *, limit: int = 3,
                history: list[dict] | None = None, write_back: bool = True,
                source: str = "") -> TurnResult:
    """One full turn, as in the book's chat loop: retrieve, answer, then save the turn to memory."""
    ret = retrieve(embedder, store, query, limit)
    system_prompt = prompts.answer_system_prompt([h.item.text for h in ret.selected])
    ans = answer(llm, query, system_prompt, history)
    write = None
    if write_back:
        turn = [{"role": "user", "content": query}, {"role": "assistant", "content": ans.reply.text}]
        write = add_conversation(llm, embedder, store, turn, source)
    return TurnResult(ret, ans, write)
