import numpy as np

from memory_lab import store as store_mod
from memory_lab.store import MemoryStore


def test_add_update_delete_are_logged(emb):
    s = MemoryStore("u")
    a = s.add("Likes tea", emb.embed(["Likes tea"])[0], "test")
    s.update(a.id, "Loves green tea", emb.embed(["Loves green tea"])[0], "test")
    assert s.get(a.id).text == "Loves green tea"
    s.delete(a.id, "test")
    assert s.memories == []
    assert [(h.event, h.old_text, h.new_text) for h in s.history] == [
        ("ADD", None, "Likes tea"), ("UPDATE", "Likes tea", "Loves green tea"), ("DELETE", "Loves green tea", None)]


def test_search_ranks_by_similarity(emb):
    s = MemoryStore("u")
    for t in ["Hikes in Marin County", "Favorite author is Isaac Asimov", "Learning guitar"]:
        s.add(t, emb.embed([t])[0])
    hits = s.search(emb.embed(["Who is my favorite author"])[0], limit=2)
    assert hits[0].item.text == "Favorite author is Isaac Asimov"
    assert len(hits) == 2 and hits[0].score >= hits[1].score


def test_threshold_filters(emb):
    s = MemoryStore("u")
    s.add("Learning guitar", emb.embed(["Learning guitar"])[0])
    assert s.search(emb.embed(["zzz qqq"])[0], threshold=0.5) == []


def test_save_and_load_round_trip(tmp_path, monkeypatch, emb):
    monkeypatch.setattr(store_mod, "STORES_DIR", tmp_path)
    s = MemoryStore("alice", built_with="fake")
    s.add("Likes tea", emb.embed(["Likes tea"])[0])
    s.save()
    loaded = store_mod.load_store("alice")
    assert [m.text for m in loaded.memories] == ["Likes tea"]
    assert np.allclose(loaded.memories[0].embedding, s.memories[0].embedding)
    assert len(loaded.history) == 1


def test_users_are_isolated(tmp_path, monkeypatch, emb):
    monkeypatch.setattr(store_mod, "STORES_DIR", tmp_path)
    a = MemoryStore("alice")
    a.add("Likes tea", emb.embed(["Likes tea"])[0])
    a.save()
    assert store_mod.load_store("bob").memories == []


def test_demo_user_starts_from_snapshot(tmp_path, monkeypatch, emb):
    snap = MemoryStore("demo_user")
    snap.add("From snapshot", emb.embed(["From snapshot"])[0])
    snap_path = tmp_path / "snap.json"
    snap.save(snap_path)
    monkeypatch.setattr(store_mod, "STORES_DIR", tmp_path / "stores")
    monkeypatch.setattr(store_mod, "DEMO_SNAPSHOT", snap_path)
    assert [m.text for m in store_mod.load_store("demo_user").memories] == ["From snapshot"]
    assert store_mod.load_store("someone_else").memories == []


def test_user_id_is_made_filename_safe():
    assert store_mod.store_path("../evil/x").name == ".._evil_x.json"


def test_manual_delete_is_logged_with_its_source(emb):
    s = MemoryStore("u")
    a = s.add("Lives in Portland", emb.embed(["Lives in Portland"])[0])
    s.delete(a.id, "Manual delete")
    assert s.history[-1].event == "DELETE" and s.history[-1].source == "Manual delete"
