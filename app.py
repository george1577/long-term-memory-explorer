"""Long-Term Memory Explorer: see how an LLM app remembers a user across conversations.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from memory_lab import data, pipeline, prompts
from memory_lab.demo import build_demo_store
from memory_lab.embedder import MiniLMEmbedder
from memory_lab.llm import DEFAULT_MODEL, MODELS, find_openai_key, make_llm, ollama_models
from memory_lab.store import DEMO_SNAPSHOT, DEMO_USER, MemoryStore, load_store, reset_to_snapshot
from memory_lab.tokens import count_message_tokens, count_tokens

st.set_page_config(page_title="Long-Term Memory Explorer", page_icon="🧠", layout="wide")

EVENT_ICON = {"ADD": "🟢 ADD", "UPDATE": "🟡 UPDATE", "DELETE": "🔴 DELETE", "NONE": "⚪ NONE"}


# =============================================================================================
# Shared resources and state
# =============================================================================================
@st.cache_resource(show_spinner="Loading the embedding model (all-MiniLM-L6-v2)...")
def get_embedder() -> MiniLMEmbedder:
    return MiniLMEmbedder()


def get_store() -> MemoryStore:
    """The current user's memory store, kept in session state and saved to disk after every change."""
    uid = st.session_state.user_id
    if st.session_state.get("store_user") != uid:
        st.session_state.store = load_store(uid)
        st.session_state.store_user = uid
        st.session_state.chat = []
    return st.session_state.store


def save_store() -> None:
    st.session_state.store.save()


def get_llm():
    label = st.session_state.model_label
    try:
        return make_llm(label, st.session_state.get("openai_key"))
    except ValueError as e:
        st.error(f"{e} Add one in the sidebar, or pick an Ollama model.")
        st.stop()


def run_llm(fn, *args, **kwargs):
    """Run a pipeline function and show a friendly message if the model can't be reached."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - we want to show any connection/model error to the learner
        st.error(f"The model call failed: {type(e).__name__}: {e}")
        if MODELS[st.session_state.model_label][0] == "ollama":
            st.info("Is Ollama running? Start the Ollama app (or `ollama serve`) and make sure the model is pulled: "
                    f"`ollama pull {MODELS[st.session_state.model_label][1]}`")
        st.stop()


# =============================================================================================
# Small rendering helpers
# =============================================================================================
def show_conversation(messages: list[dict], height: int | None = 380) -> None:
    box = st.container(height=height) if height else st.container()
    with box:
        for m in messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])


def reply_caption(reply, label: str = "") -> None:
    st.caption(f"{label}{reply.prompt_tokens:,} tokens in · {reply.completion_tokens:,} tokens out · "
               f"{reply.seconds:.1f} s")


def store_table(store: MemoryStore) -> pd.DataFrame:
    return pd.DataFrame(
        [{"memory": m.text, "id": m.id[:8], "updated": m.updated_at.replace("T", " ")[:19]} for m in store.memories],
        columns=["memory", "id", "updated"],
    )


def history_table(store: MemoryStore) -> pd.DataFrame:
    rows = [{"when": h.time.replace("T", " ")[:19], "event": EVENT_ICON.get(h.event, h.event),
             "before": h.old_text or "", "after": h.new_text or "", "caused by": h.source, "id": h.memory_id[:8]}
            for h in reversed(store.history)]
    return pd.DataFrame(rows, columns=["when", "event", "before", "after", "caused by", "id"])


def decisions_table(decisions: list[pipeline.Decision]) -> pd.DataFrame:
    rows = [{"decision": EVENT_ICON.get(d.event, d.event), "model's id": d.llm_id, "text": d.text,
             "old text": d.old_text or "", "applied?": "✅" if d.applied else ("—" if d.event == "NONE" else "❌"),
             "note": d.note} for d in decisions]
    return pd.DataFrame(rows, columns=["decision", "model's id", "text", "old text", "applied?", "note"])


def show_extraction(ex: pipeline.ExtractionResult) -> None:
    if ex.facts:
        st.markdown("**Candidate facts the model pulled out:**")
        for f in ex.facts:
            st.markdown(f"- {f}")
    elif ex.parsed:
        st.info("The model found **no facts worth remembering** (it returned an empty list).")
    else:
        st.warning("The model's reply was not valid JSON, so (like mem0) the pipeline treats it as no facts.")
    reply_caption(ex.reply, "Extraction call: ")
    with st.expander("See the exact prompt and the model's raw reply"):
        st.markdown("**System prompt** (mem0's fact-extraction prompt):")
        st.code(ex.system_prompt, language="text", wrap_lines=True)
        st.markdown("**User message** (the conversation, flattened):")
        st.code(ex.user_message, language="text", wrap_lines=True)
        st.markdown("**Raw reply:**")
        st.code(ex.reply.text, language="json", wrap_lines=True)


def show_reconciliation(rec: pipeline.ReconciliationResult, store_before: list[str] | None = None) -> None:
    if not rec.facts:
        st.info("No candidate facts, so reconciliation is skipped entirely (no LLM call), just like mem0.")
        return
    st.markdown("**Step A: find similar existing memories for each fact** "
                f"(top {pipeline.CANDIDATES_PER_FACT} per fact by similarity score)")
    if rec.candidates:
        df = pd.DataFrame([{"candidate fact": c.fact, "similar existing memory": c.text, "score": round(c.score, 3)}
                           for c in rec.candidates])
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("The store is empty, so there is nothing to compare against. Every fact should become an ADD.")
    st.markdown("**Step B: give each existing memory a short id** (mem0 does this because models garble long ids)")
    if rec.id_map:
        st.dataframe(pd.DataFrame([{"short id shown to model": k, "real memory id": v[:8] + "…",
                                    "text": next(o["text"] for o in rec.old_memory if o["id"] == k)}
                                   for k, v in rec.id_map.items()]), hide_index=True, width="stretch")
    st.markdown("**Step C: the model decides ADD / UPDATE / DELETE / NONE**")
    if rec.decisions:
        st.dataframe(decisions_table(rec.decisions), hide_index=True, width="stretch")
    elif not rec.parsed:
        st.warning("The model's reply was not valid JSON, so no changes were made.")
    else:
        st.info("The model returned no decisions.")
    if rec.reply:
        reply_caption(rec.reply, "Reconciliation call: ")
    with st.expander("See the exact reconciliation prompt and the model's raw reply"):
        st.code(rec.prompt or "", language="text", wrap_lines=True)
        st.markdown("**Raw reply:**")
        st.code(rec.reply.text if rec.reply else "", language="json", wrap_lines=True)


def score_chart(ranked: list, selected: list) -> alt.Chart:
    chosen = {id(h) for h in selected}
    df = pd.DataFrame([{"memory": h.item.text, "score": round(h.score, 3),
                        "retrieved": "retrieved" if id(h) in chosen else "not retrieved"} for h in ranked])
    return (alt.Chart(df).mark_bar().encode(
        x=alt.X("score:Q", title="similarity score", scale=alt.Scale(domain=[min(0, df.score.min()), 1])),
        y=alt.Y("memory:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420)),
        color=alt.Color("retrieved:N", scale=alt.Scale(domain=["retrieved", "not retrieved"],
                                                         range=["#2e7d32", "#c8c8c8"]), legend=alt.Legend(title=None)),
        tooltip=["memory", "score"],
    ).properties(height=max(120, 24 * len(df))))


# =============================================================================================
# Sidebar
# =============================================================================================
with st.sidebar:
    st.header("🧠 Long-Term Memory Explorer")
    st.caption("Companion to *Generative AI Design Patterns*, pattern 28 (Long-Term Memory).")

    st.selectbox("Model", list(MODELS), index=list(MODELS).index(DEFAULT_MODEL), key="model_label",
                 help="Used for extraction, reconciliation and answering.")
    provider, model_name = MODELS[st.session_state.model_label]
    if provider == "ollama":
        pulled = ollama_models()
        if pulled is None:
            st.error("Ollama isn't reachable. Start the Ollama app or run `ollama serve`.")
        elif model_name not in pulled:
            st.warning(f"`{model_name}` isn't pulled yet. Run `ollama pull {model_name}`.")
        else:
            st.success(f"Ollama is running, `{model_name}` is ready.")
    else:
        if "openai_key" not in st.session_state:
            st.session_state.openai_key = find_openai_key() or ""
        st.text_input("OpenAI API key", key="openai_key", type="password",
                      help="Found automatically in OPENAI_API_KEY, ./.env, or the book repo's examples/keys.env.")
        st.caption("Calls to gpt-4o-mini cost a little money (a fraction of a cent per call).")

    st.divider()
    st.text_input("User id", value=DEMO_USER, key="user_id",
                  help="Each user has a separate memory store. `demo_user` starts with the memories built "
                       "from the notebook's 5 conversations. Any other id starts empty.")
    store = get_store()
    st.metric("Memories in this user's store", len(store.memories))
    if store.built_with:
        st.caption(f"Store first built with `{store.built_with}`.")
    c1, c2 = st.columns(2)
    if c1.button("Reset to demo", width="stretch", disabled=not DEMO_SNAPSHOT.exists(),
                 help="Replace this user's memories with the shipped demo snapshot."):
        st.session_state.store = reset_to_snapshot(st.session_state.user_id)
        st.rerun()
    if c2.button("Empty store", width="stretch", help="Delete every memory for this user."):
        store.clear()
        save_store()
        st.rerun()
    st.download_button("Download store (JSON)", json.dumps(store.to_dict(), indent=1),
                       file_name=f"{store.user_id}_memory_store.json", width="stretch")
    st.caption("The store is a plain JSON file in `data/stores/`. It survives app restarts: that's what makes "
               "it *long-term*.")


# =============================================================================================
# Tabs
# =============================================================================================
st.title("Long-Term Memory Explorer")
st.markdown("An LLM forgets everything between calls. **Long-term memory** fixes that: after each conversation, "
            "pull out the facts worth keeping, store them, and bring back the relevant ones next time. "
            "The tabs below go through it one step at a time. New to this? Read "
            "`docs/how-long-term-memory-works.md`.")

tabs = st.tabs(["① Why memory?", "② Extraction", "③ Reconciliation", "④ Retrieval",
                "⑤ Answering with memory", "⑥ Chat", "⑦ Real mem0"])

# ---------------------------------------------------------------------------------------------
# ① Why memory?
# ---------------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("① Why does an LLM need memory at all?")
    st.markdown("Ask the same question **three ways**: with no context, with every past conversation pasted in "
                "full, and with a few memories retrieved from the store. Compare the answers *and* the size of "
                "the prompt.")
    q_choice = st.selectbox("Question", data.queries() + ["(write your own)"], index=1, key="t1_q")
    query = st.text_input("Your question", key="t1_custom") if q_choice == "(write your own)" else q_choice
    if st.button("Ask three ways", type="primary", key="t1_go", disabled=not query):
        llm, emb, store = get_llm(), get_embedder(), get_store()
        convs = [c["messages"] for c in data.conversations()]
        ret = pipeline.retrieve(emb, store, query, limit=3)
        systems = {
            "no_memory": prompts.answer_system_prompt([]),
            "transcript": prompts.transcript_system_prompt(convs),
            "memory": prompts.answer_system_prompt([h.item.text for h in ret.selected]),
        }
        with st.spinner("Asking the model three times..."):
            st.session_state.t1 = {k: run_llm(pipeline.answer, llm, query, s) for k, s in systems.items()}
            st.session_state.t1_ret = ret

    if "t1" in st.session_state:
        res = st.session_state.t1
        cols = st.columns(3)
        titles = {"no_memory": "🙈 No memory", "transcript": "📜 Paste every past conversation",
                  "memory": "🧠 Long-term memory (top 3)"}
        for col, (k, r) in zip(cols, res.items()):
            with col:
                st.markdown(f"#### {titles[k]}")
                st.metric("Prompt size", f"{count_message_tokens(r.messages):,} tokens")
                st.write(r.reply.text)
                reply_caption(r.reply)
                with st.expander("Messages sent"):
                    st.code(r.messages[0]["content"], language="text", wrap_lines=True)
        st.markdown("#### What happens as conversations pile up?")
        per_conv = count_tokens(prompts.transcript_system_prompt([c["messages"] for c in data.conversations()])) / 5
        mem_tokens = count_tokens(res["memory"].messages[0]["content"])
        rows = []
        for n in [5, 50, 100, 500, 1000]:
            rows += [{"past conversations": n, "approach": "paste every conversation", "prompt tokens": int(per_conv * n)},
                     {"past conversations": n, "approach": "long-term memory (top 3)", "prompt tokens": mem_tokens}]
        chart = alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
            x="past conversations:Q", y=alt.Y("prompt tokens:Q", scale=alt.Scale(type="log")), color="approach:N")
        st.altair_chart(chart, width="stretch")
        st.caption(f"Each sample conversation is about {per_conv:,.0f} tokens. Pasting everything grows without "
                   "limit and soon overflows the context window (128k tokens for gpt-4o-mini, and we give the local "
                   "models 8k). Retrieved memories stay a few dozen tokens no matter how much history there is. "
                   "The y-axis is logarithmic.")

# ---------------------------------------------------------------------------------------------
# ② Extraction
# ---------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("② Extraction: which facts are worth remembering?")
    st.markdown("The first half of the **write path**. An LLM reads a conversation and returns a short list of "
                "facts about the user, skipping small talk and trivia. This tab only *shows* the facts: "
                "nothing is saved to the store yet (that's tab ③).")
    convs = data.conversations()
    options = [f"{c['id']}. {c['title']}" for c in convs] + ["Write your own"]
    pick = st.radio("Conversation", options, horizontal=True, key="t2_pick")
    if pick == "Write your own":
        raw = st.text_area("One message per line, starting with `user:` or `assistant:`",
                           "user: Hi! I'm Sam and I'm allergic to peanuts.\n"
                           "assistant: Thanks for telling me, Sam. I'll keep that in mind.\n"
                           "user: What's the capital of France?\nassistant: Paris.", height=150, key="t2_raw")
        messages = []
        for line in raw.splitlines():
            role, _, content = line.partition(":")
            if role.strip().lower() in ("user", "assistant") and content.strip():
                messages.append({"role": role.strip().lower(), "content": content.strip()})
        conv = None
    else:
        conv = convs[options.index(pick)]
        messages = conv["messages"]
    left, right = st.columns([1, 1])
    with left:
        if conv:
            st.info(f"**What to watch for:** {conv['trap']}")
        show_conversation(messages)
    with right:
        if st.button("Extract facts", type="primary", key="t2_go", disabled=not messages):
            with st.spinner("Extracting..."):
                st.session_state.t2 = (pick, run_llm(pipeline.extract, get_llm(), messages))
        if st.session_state.get("t2") and st.session_state.t2[0] == pick:
            show_extraction(st.session_state.t2[1])

# ---------------------------------------------------------------------------------------------
# ③ Reconciliation
# ---------------------------------------------------------------------------------------------
PRESETS = {
    "Correct the location trap": "Actually, I live in Portland, Oregon, not San Francisco. I work there as a software engineer.",
    "Move and change hobbies": "Big news: I just moved to Seattle. I've given up hiking and I play tennis now.",
    "Repeat something known": "I really like Isaac Asimov's Foundation books.",
    "Share something new": "My name is Alex, and I'm vegetarian.",
    "Small talk only": "What's the tallest mountain in the world?",
}
with tabs[2]:
    st.subheader("③ Reconciliation: how new facts change existing memories")
    st.markdown("The second half of the **write path**. New candidate facts are compared with similar memories "
                "already in the store, and the model picks one decision for each: **ADD** a new memory, "
                "**UPDATE** an old one, **DELETE** one that's now wrong, or **NONE** (nothing to change). "
                "These changes are **saved** to the current user's store.")
    store = get_store()
    with st.expander(f"Current memory store for `{store.user_id}` ({len(store.memories)} memories)", expanded=True):
        st.dataframe(store_table(store), hide_index=True, width="stretch")
    preset = st.radio("Try a preset", list(PRESETS), horizontal=True, key="t3_preset")
    msg = st.text_area("Something the user says (treated as a one-message conversation)", PRESETS[preset],
                       key=f"t3_msg_{preset}", height=80)
    if st.button("Extract and reconcile", type="primary", key="t3_go", disabled=not msg.strip()):
        before = [m.text for m in store.memories]
        with st.spinner("Extracting, then reconciling..."):
            res = run_llm(pipeline.add_conversation, get_llm(), get_embedder(), store,
                          [{"role": "user", "content": msg.strip()}], source="Tab ③")
        save_store()
        st.session_state.t3 = (res, before)
    if "t3" in st.session_state:
        res, before = st.session_state.t3
        st.markdown("#### 1. Extraction")
        show_extraction(res.extraction)
        st.markdown("#### 2. Reconciliation")
        show_reconciliation(res.reconciliation)
        st.markdown("#### 3. The store, before and after")
        after = [m.text for m in get_store().memories]
        b, a = st.columns(2)
        b.markdown("**Before**")
        b.dataframe(pd.DataFrame({"memory": before}), hide_index=True, width="stretch")
        a.markdown("**After**")
        a.dataframe(pd.DataFrame({"memory": after, "change": ["🆕 new or changed" if t not in before else ""
                                                             for t in after]}),
                    hide_index=True, width="stretch")
    with st.expander("History log (every change ever made to this store)"):
        st.dataframe(history_table(get_store()), hide_index=True, width="stretch")
    with st.expander("Where did the demo store come from? Rebuild it from the 5 conversations"):
        st.markdown("The shipped snapshot was made by running extraction and reconciliation on the notebook's 5 "
                    "sample conversations, in order, starting from an empty store. You can do it again with the "
                    "model chosen in the sidebar. It takes a few minutes on a laptop, and it **replaces** the current "
                    "user's memories.")
        if st.button("Rebuild from the 5 conversations", key="t3_rebuild"):
            log = st.container()

            def on_step(i, conv, r):
                with log:
                    st.markdown(f"**Conversation {i}: {conv['title']}**")
                    st.write("Facts:", r.extraction.facts or "(none)")
                    if r.reconciliation.decisions:
                        st.dataframe(decisions_table(r.reconciliation.decisions), hide_index=True,
                                     width="stretch")

            with st.spinner("Building... (each conversation takes two LLM calls)"):
                new_store, _ = run_llm(build_demo_store, get_llm(), get_embedder(), st.session_state.user_id, on_step)
            st.session_state.store = new_store
            save_store()
            st.success(f"Done: {len(new_store.memories)} memories. The sidebar count updates on the next click.")

# ---------------------------------------------------------------------------------------------
# ④ Retrieval
# ---------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("④ Retrieval: finding the right memories for a question")
    st.markdown("The **read path**. The question and every memory are turned into embeddings (lists of numbers that "
                "capture meaning). Memories whose meaning is closest to the question get the highest "
                "**similarity score**, and the top few are retrieved. No LLM is involved, which is why this is instant.")
    store = get_store()
    if not store.memories:
        st.warning("This user's store is empty. Use **Reset to demo** in the sidebar, or add memories in tab ③.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        q_choice = c1.selectbox("Question", data.queries() + ["(write your own)"], key="t4_q")
        query = c1.text_input("Your question", key="t4_custom") if q_choice == "(write your own)" else q_choice
        limit = c2.slider("limit (how many to retrieve)", 1, 10, 2, key="t4_limit",
                          help="The notebook's demo uses limit=2; its chat loop uses 3.")
        use_thr = c3.checkbox("Minimum score", key="t4_use_thr", help="mem0's `threshold` option.")
        thr = c3.slider("threshold", 0.0, 1.0, 0.3, 0.05, key="t4_thr", disabled=not use_thr)
        if query:
            ret = pipeline.retrieve(get_embedder(), store, query, limit, thr if use_thr else None)
            st.altair_chart(score_chart(ret.ranked, ret.selected), width="stretch")
            st.markdown("**Retrieved:** " + (", ".join(f"*{h.item.text}* ({h.score:.2f})" for h in ret.selected)
                                             or "nothing passed the threshold"))
            if ret.selected and ret.selected[0].score < 0.3:
                st.warning("The best score is low: **nothing in the store is really about this question.** Retrieval "
                           "still returns the top results anyway unless you set a minimum score. Try "
                           "*What's my name?* (no name was ever given).")
        st.markdown("#### All 5 notebook queries at once")
        all_rows = []
        for q in data.queries():
            r = pipeline.retrieve(get_embedder(), store, q, limit)
            for rank, h in enumerate(r.selected, 1):
                all_rows.append({"query": q, "rank": rank, "memory": h.item.text, "score": round(h.score, 3)})
        st.dataframe(pd.DataFrame(all_rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
# ⑤ Answering with memory
# ---------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("⑤ Answering with memory: the full loop")
    st.markdown("This is one turn of the book's `chat_with_memory` loop: **retrieve** memories → build a "
                "**memory-augmented prompt** → **answer** → **write the turn back** to memory (extraction + "
                "reconciliation).")
    c1, c2, c3 = st.columns([2, 1, 1])
    q_choice = c1.selectbox("Question", data.queries() + ["(write your own)"], index=3, key="t5_q")
    query = c1.text_input("Your question", key="t5_custom") if q_choice == "(write your own)" else q_choice
    limit = c2.slider("limit", 1, 10, 3, key="t5_limit")
    write_back = c3.checkbox("Write the turn back to memory", value=True, key="t5_write")
    if st.button("Run one turn", type="primary", key="t5_go", disabled=not query):
        store = get_store()
        with st.spinner("Retrieving, answering" + (", then writing back..." if write_back else "...")):
            st.session_state.t5 = run_llm(pipeline.memory_turn, get_llm(), get_embedder(), store, query,
                                          limit=limit, write_back=write_back, source="Tab ⑤")
        save_store()
    if "t5" in st.session_state:
        t = st.session_state.t5
        st.markdown("#### Step 1: Retrieve")
        st.dataframe(pd.DataFrame([{"memory": h.item.text, "score": round(h.score, 3)} for h in t.retrieval.selected]),
                     hide_index=True, width="stretch")
        st.markdown("#### Step 2: Build the memory-augmented prompt")
        st.code(t.answer.messages[0]["content"], language="text", wrap_lines=True)
        st.markdown("#### Step 3: Answer")
        with st.chat_message("user"):
            st.write(t.retrieval.query)
        with st.chat_message("assistant"):
            st.write(t.answer.reply.text)
        reply_caption(t.answer.reply, "Answer call: ")
        st.markdown("#### Step 4: Write the turn back")
        if t.write is None:
            st.caption("Skipped (write-back is off).")
        else:
            show_extraction(t.write.extraction)
            show_reconciliation(t.write.reconciliation)

# ---------------------------------------------------------------------------------------------
# ⑥ Chat
# ---------------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("⑥ Chat: short-term vs long-term memory")
    st.markdown("Chat freely. Each turn runs the full loop from tab ⑤. The **conversation history** (short-term) "
                "lives only in this chat; the **memory store** (long-term) is on the right. Click "
                "**Start a new conversation**: the chat is wiped, but the memories stay. Then ask what it remembers.")
    store = get_store()
    left, right = st.columns([3, 2])
    with left:
        b1, b2, b3 = st.columns([2, 1, 1])
        if b1.button("🆕 Start a new conversation (memories stay)", key="t6_new"):
            st.session_state.chat = []
        send_history = b2.checkbox("Send chat history", value=True, key="t6_hist",
                                   help="Include earlier messages of this chat in the prompt (short-term memory). "
                                        "The book's example doesn't; turn it off to match it exactly.")
        chat_limit = b3.number_input("limit", 1, 10, 3, key="t6_limit")
        chat_box = st.container(height=480)
        with chat_box:
            for turn in st.session_state.get("chat", []):
                with st.chat_message(turn["role"]):
                    st.write(turn["content"])
                    if turn.get("details"):
                        d = turn["details"]
                        with st.expander("What memory did this turn use and change?"):
                            st.markdown("**Retrieved:** " + ("; ".join(f"{t} ({s:.2f})" for t, s in d["retrieved"])
                                                             or "nothing"))
                            st.markdown("**Facts extracted from this turn:** " + ("; ".join(d["facts"]) or "none"))
                            if d["decisions"]:
                                st.dataframe(pd.DataFrame(d["decisions"]), hide_index=True, width="stretch")
        user_msg = st.chat_input("Say something about yourself, or ask what I remember...", key="t6_input")
        if user_msg:
            history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat] if send_history else []
            with st.spinner("Thinking, then updating memory..."):
                t = run_llm(pipeline.memory_turn, get_llm(), get_embedder(), store, user_msg, limit=int(chat_limit),
                            history=history, source="Chat")
            save_store()
            st.session_state.chat.append({"role": "user", "content": user_msg})
            st.session_state.chat.append({"role": "assistant", "content": t.answer.reply.text, "details": {
                "retrieved": [(h.item.text, h.score) for h in t.retrieval.selected],
                "facts": t.write.extraction.facts if t.write else [],
                "decisions": [{"decision": EVENT_ICON.get(d.event, d.event), "text": d.text,
                               "applied": d.applied} for d in (t.write.reconciliation.decisions if t.write else [])],
            }})
            st.rerun()
    with right:
        st.markdown(f"**Memory store for `{store.user_id}`** ({len(store.memories)} memories)")
        st.dataframe(store_table(store)[["memory"]], hide_index=True, width="stretch", height=300)
        st.markdown("**Recent changes**")
        st.dataframe(history_table(store)[["event", "before", "after", "caused by"]].head(10), hide_index=True,
                     width="stretch")

# ---------------------------------------------------------------------------------------------
# ⑦ Real mem0
# ---------------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("⑦ The real thing: mem0")
    st.markdown("Everything above is a transparent re-implementation of what the **mem0** library does inside "
                "`memory.add()` and `memory.search()`, using mem0's own prompts. Here we run the real `mem0ai==0.1.113` "
                "(the version the book pins) on the same 5 conversations, with the model from the sidebar, the same "
                "embedding model, and a fresh temporary Chroma database. Then we compare.")
    st.caption("Takes a few minutes with a local model (10 LLM calls). mem0's telemetry is switched off.")
    m_limit = st.slider("search limit", 1, 5, 2, key="t7_limit")
    if st.button("Run mem0 on the 5 conversations", type="primary", key="t7_go"):
        from memory_lab.mem0_runner import run_mem0

        status = st.status("Running mem0...", expanded=True)
        with status:
            run = run_llm(run_mem0, data.conversations(), data.queries(), st.session_state.model_label,
                          st.session_state.get("openai_key"), limit=m_limit, on_step=st.write)
        status.update(label=f"mem0 finished in {run.seconds:.0f} s", state="complete", expanded=False)
        st.session_state.t7 = (st.session_state.model_label, run)
    if "t7" in st.session_state:
        label, run = st.session_state.t7
        st.caption(f"Run with **{label}**.")
        st.markdown("#### What `memory.add()` returned for each conversation")
        rows = []
        for conv, evs in zip(data.conversations(), run.events):
            if not evs:
                rows.append({"conversation": f"{conv['id']}. {conv['title']}", "event": "(nothing)", "memory": ""})
            for e in evs:
                rows.append({"conversation": f"{conv['id']}. {conv['title']}",
                             "event": EVENT_ICON.get(e.get("event"), e.get("event")), "memory": e.get("memory", "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("#### Final memories: mem0 vs this app's engine")
        c1, c2 = st.columns(2)
        c1.markdown(f"**mem0** ({len(run.memories)} memories)")
        c1.dataframe(pd.DataFrame({"memory": run.memories}), hide_index=True, width="stretch")
        ours = load_store(DEMO_USER) if st.session_state.user_id != DEMO_USER else get_store()
        snap = MemoryStore.from_dict(json.loads(DEMO_SNAPSHOT.read_text())) if DEMO_SNAPSHOT.exists() else ours
        c2.markdown(f"**This app's engine**: the shipped demo snapshot ({len(snap.memories)} memories, "
                    f"built with `{snap.built_with}`)")
        c2.dataframe(store_table(snap)[["memory"]], hide_index=True, width="stretch")
        st.caption("They won't match word for word, even with the same model: LLM output varies from run to run. "
                   "What should match is the *kind* of thing kept and dropped.")

        st.markdown("#### `memory.search()` results")
        srows = [{"query": s.query, "memory": r["memory"], "mem0 'score' (a distance!)": round(r["distance"], 3),
                  "as cosine similarity": round(r["cosine"], 3)} for s in run.searches for r in s.results]
        st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch")
        st.warning("**Gotcha:** with Chroma, the number mem0 calls `score` is a *distance*: **lower means more "
                   "relevant**. That is the opposite of this app's similarity score. Because the embeddings have "
                   "length 1, you can convert: cosine similarity = 1 − distance ÷ 2. If you set mem0's `threshold` "
                   "option expecting 'higher is better', you'd filter out the *best* matches.")
