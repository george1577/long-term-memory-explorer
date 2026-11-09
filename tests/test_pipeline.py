from memory_lab import pipeline
from memory_lab.store import MemoryStore
from tests.conftest import FakeLLM


def test_extract_returns_facts():
    llm = FakeLLM({"facts": ["Is a software engineer", "Lives in Portland"]})
    ex = pipeline.extract(llm, [{"role": "user", "content": "I'm a software engineer in Portland"}])
    assert ex.facts == ["Is a software engineer", "Lives in Portland"]
    assert ex.parsed
    assert llm.calls[0][0]["role"] == "system" and "user: I'm a software engineer" in llm.calls[0][1]["content"]


def test_extract_with_bad_json_gives_no_facts():
    ex = pipeline.extract(FakeLLM("I think the user likes tea"), [{"role": "user", "content": "hi"}])
    assert ex.facts == [] and not ex.parsed


def test_no_facts_means_no_reconciliation_call(emb):
    llm = FakeLLM()  # would raise if called
    rec = pipeline.reconcile(llm, emb, MemoryStore("u"), [])
    assert rec.prompt is None and rec.decisions == [] and llm.calls == []


def test_empty_store_facts_are_added(emb):
    s = MemoryStore("u")
    llm = FakeLLM({"memory": [{"id": "0", "text": "Likes tea", "event": "ADD"}]})
    rec = pipeline.reconcile(llm, emb, s, ["Likes tea"])
    assert [m.text for m in s.memories] == ["Likes tea"]
    assert rec.old_memory == [] and rec.counts()["ADD"] == 1


def test_short_ids_map_back_to_real_ids_for_update_and_delete(emb):
    s = MemoryStore("u")
    city = s.add("Lives in San Francisco", emb.embed(["Lives in San Francisco"])[0])
    tea = s.add("Likes tea", emb.embed(["Likes tea"])[0])
    facts = ["Lives in Portland", "Dislikes tea"]
    # Work out which short id the pipeline will give each memory, then script the model's answer.
    probe = pipeline.reconcile(FakeLLM({"memory": []}), emb, s, facts)
    short = {s.get(real).text: k for k, real in probe.id_map.items()}
    llm = FakeLLM({"memory": [
        {"id": short["Lives in San Francisco"], "text": "Lives in Portland", "event": "UPDATE",
         "old_memory": "Lives in San Francisco"},
        {"id": short["Likes tea"], "text": "Likes tea", "event": "DELETE"},
    ]})
    rec = pipeline.reconcile(llm, emb, s, facts, source="test")
    assert [m.text for m in s.memories] == ["Lives in Portland"]
    assert s.memories[0].id == city.id  # UPDATE keeps the same memory id
    assert s.get(tea.id) is None
    assert all(d.applied for d in rec.decisions)
    # The model was shown short ids, never the real UUIDs.
    assert city.id not in rec.prompt and tea.id not in rec.prompt


def test_invented_id_is_skipped_not_crashing(emb):
    s = MemoryStore("u")
    s.add("Likes tea", emb.embed(["Likes tea"])[0])
    llm = FakeLLM({"memory": [{"id": "7", "text": "Loves tea", "event": "UPDATE"}]})
    rec = pipeline.reconcile(llm, emb, s, ["Loves tea"])
    assert not rec.decisions[0].applied and "not one of the memories" in rec.decisions[0].note
    assert [m.text for m in s.memories] == ["Likes tea"]


def test_none_changes_nothing(emb):
    s = MemoryStore("u")
    s.add("Likes tea", emb.embed(["Likes tea"])[0])
    llm = FakeLLM({"memory": [{"id": "0", "text": "Likes tea", "event": "NONE"}]})
    rec = pipeline.reconcile(llm, emb, s, ["Likes tea"])
    assert len(s.memories) == 1 and len(s.history) == 1 and rec.counts()["NONE"] == 1


def test_retrieve_respects_limit_and_threshold(emb):
    s = MemoryStore("u")
    for t in ["Favorite author is Isaac Asimov", "Learning guitar", "Hikes in Marin County"]:
        s.add(t, emb.embed([t])[0])
    r = pipeline.retrieve(emb, s, "favorite author", limit=1)
    assert len(r.ranked) == 3 and [h.item.text for h in r.selected] == ["Favorite author is Isaac Asimov"]
    assert pipeline.retrieve(emb, s, "favorite author", limit=3, threshold=0.99).selected == []


def test_memory_turn_retrieves_answers_and_writes_back(emb):
    s = MemoryStore("u")
    s.add("Favorite author is Isaac Asimov", emb.embed(["Favorite author is Isaac Asimov"])[0])
    llm = FakeLLM(
        "You love Asimov!",                                              # answer
        {"facts": ["Wants book recommendations"]},                      # extraction of the turn
        {"memory": [{"id": "1", "text": "Wants book recommendations", "event": "ADD"}]},  # reconciliation
    )
    t = pipeline.memory_turn(llm, emb, s, "Who is my favorite author?", limit=1)
    assert "- Favorite author is Isaac Asimov" in t.answer.messages[0]["content"]
    assert t.answer.reply.text == "You love Asimov!"
    assert "Wants book recommendations" in [m.text for m in s.memories]
    # The turn written back contains both the question and the answer.
    assert "assistant: You love Asimov!" in llm.calls[1][1]["content"]


def test_memory_turn_includes_chat_history(emb):
    llm = FakeLLM("ok")
    hist = [{"role": "user", "content": "I'm Sam"}, {"role": "assistant", "content": "Hi Sam"}]
    t = pipeline.memory_turn(llm, emb, MemoryStore("u"), "What's my name?", history=hist, write_back=False)
    assert [m["role"] for m in t.answer.messages] == ["system", "user", "assistant", "user"]
    assert t.write is None
