# Long-Term Memory Explorer

A learning tool for seeing how an LLM application remembers facts about a user across conversations: what gets saved, how saved facts change, and how they are found again.

## Language

### What is remembered

**Memory**:
One short fact about the user, written as a standalone sentence, e.g. "Is a software engineer in Portland".
_Avoid_: Note, record, entry, embedding

**Memory store**:
The complete set of memories kept for one user, which persists after a conversation ends.
_Avoid_: Database, vector store, memory (for the whole collection)

**Conversation history**:
The list of messages in the current conversation; short-term context that is not part of the memory store.
_Avoid_: Memory, short-term memory, chat log

**Conversation**:
An ordered list of user and assistant messages that is handed to the memory pipeline as one unit.
_Avoid_: Session, thread, dialogue

### The memory pipeline

**Extraction**:
The step where an LLM reads a conversation and pulls out the facts about the user that are worth remembering, ignoring small talk.
_Avoid_: Summarization, parsing

**Candidate fact**:
A fact produced by extraction that has not yet been compared with the memory store.
_Avoid_: New memory (it may never become one)

**Reconciliation**:
The step where an LLM compares candidate facts with similar existing memories and chooses one memory decision for each.
_Avoid_: Update (that is one of the decisions), merging, deduplication

**Memory decision**:
The outcome of reconciliation for one memory: ADD (save a new memory), UPDATE (rewrite an existing memory), DELETE (remove a memory that is now wrong), or NONE (change nothing).
_Avoid_: Event, operation, action

**Retrieval**:
Finding the memories most relevant to a query by comparing their meaning, not their exact words.
_Avoid_: Lookup, recall, search (fine in casual use, but retrieval is the canonical term)

**Similarity score**:
A number showing how close a memory's meaning is to a query; higher means more relevant.
_Avoid_: Confidence, relevance probability

**Memory-augmented prompt**:
The system prompt built by inserting retrieved memories so the LLM can personalise its answer.
_Avoid_: Context injection, RAG prompt
