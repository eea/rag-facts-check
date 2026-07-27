# Architecture Tension: Multi-Turn Conversations

> When to revisit: when the chatbot integrates multi-turn fact-checking.

## The core question

The checker takes `(answer, documents)` and verifies claims in `answer` against
`documents`. In a multi-turn conversation, what counts as the "answer" and what
counts as the "documents"?

## Scenarios to consider

### 1. Final answer only (simplest)

The chatbot sends only the *last assistant message* + *documents retrieved for
that turn*. This is what the system already does. Works fine if each turn is
self-contained.

**Problem:** If the assistant builds on earlier turns ("As I mentioned, Paris is
the capital..."), claims from earlier turns are implicitly repeated but not
re-verified.

### 2. Full conversation history as answer

Send the entire conversation (all assistant messages concatenated) + all
documents from all turns.

**Problem:** The claim extractor will extract claims from *user messages too*
("Is Berlin the capital?"). User questions aren't claims to verify. You'd need
to filter by speaker role.

### 3. Per-turn fact-checking

Run the checker on each assistant turn independently, accumulate reports.

**Problem:** A claim might be supported by a document from turn 1 but the
assistant references it in turn 3. Per-turn checking would flag it as "not
enough info" because turn 3's documents don't include turn 1's sources.

### 4. Cumulative documents, latest answer (recommended)

Send the *latest assistant message* + *all documents from all turns*.

**Benefit:** The latest answer can reference any document from the full
conversation. No cross-turn gap.

**Problem:** Context window grows with conversation length. May need document
pruning or summarization.

## What the current system already handles

- The `documents` list is just strings — no assumption about turn structure
- `doc_id` lets you track which turn/source a document came from
- The checker is stateless — each call is independent

## What might need adaptation

1. **Claim extraction prompt** — currently extracts from a single text block.
   If you feed full conversation history, you'd need to mark speaker roles so it
   ignores user messages.

2. **Document deduplication** — same document might be retrieved across turns.
   The retriever would chunk it multiple times with different IDs.

3. **Context window** — `max_docs_chars=8000` is a hard limit. Long
   conversations with many documents would hit this.

## Recommendation

**Don't change the checker.** The cleanest approach is:

- **Client-side**: decide what to send. Most chatbots should send
  `(latest_assistant_message, all_retrieved_documents)` — this covers scenario 4.
- **The checker stays dumb**: it verifies claims in a text against documents. It
  doesn't need to understand conversation structure.
- **If conversation-aware checking is ever needed**, add a
  `ConversationFactsChecker` wrapper that:
  - Filters assistant messages from user messages
  - Deduplicates documents by `doc_id`
  - Runs `check()` per assistant message or on the full text
  - Merges reports

This keeps the core library simple and pushes conversation logic to the caller
where it belongs (the chatbot knows its own message format).
