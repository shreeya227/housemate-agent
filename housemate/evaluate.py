"""Measure search quality with Recall@k.

For each question we write down the message IDs that truly answer it.
Recall@3 = share of questions where at least one correct ID is in the top 3.
This is how we know search is good, instead of hoping it is.

Run:  python -m housemate.evaluate
"""
from __future__ import annotations

from pathlib import Path

from housemate.ingest import load_chat
from housemate.search import BM25Index, message_docs

DATA = Path(__file__).parent.parent / "data"

# (question, IDs of messages that answer it)
EVAL_SET = [
    ("When do the trash bins go out?", {"msg-002"}),
    ("How do bathrooms get cleaned?", {"msg-003", "msg-004"}),
    ("Can I do laundry on the weekend?", {"msg-006", "msg-007"}),
    ("Which groceries are shared?", {"msg-008"}),
    ("How often is the Costco run and how do we split it?", {"msg-009", "msg-011"}),
    ("Who drives first to Costco?", {"msg-010", "msg-011"}),
    ("What are the rules for Sunday dinner?", {"msg-013", "msg-014"}),
    ("Is the Sunday dinner vegetarian?", {"msg-013", "msg-014"}),
    ("What is the top shelf of the fridge for?", {"msg-015"}),
    ("Who takes out the garbage?", {"msg-002"}),
    # Synonym test: the chat says "trash", never "rubbish". Keyword search should miss this.
    ("Who handles the rubbish?", {"msg-002"}),
]


def recall_at_k(index: BM25Index, k: int = 3, verbose: bool = False) -> float:
    found = 0
    for question, expected in EVAL_SET:
        top_ids = [hit.doc.id for hit in index.search(question, k=k)]
        ok = bool(expected & set(top_ids))
        found += ok
        if verbose:
            print(f"{'PASS' if ok else 'MISS'}  {question}\n      expected {sorted(expected)}, got {top_ids}")
    return found / len(EVAL_SET)


if __name__ == "__main__":
    index = BM25Index(message_docs(load_chat(DATA / "house_chat.txt")))
    score = recall_at_k(index, k=3, verbose=True)
    print(f"\nRecall@3: {score:.0%}")
