"""Step 5: pull decisions and tasks out of the house chat.

The agent reads the whole chat (read_chat), then records each agreement
(record_decision) and each concrete job (add_task), always with citations.

We score it against a hand-labeled answer key, the same idea as Recall@3
in Step 2: we measure quality instead of eyeballing it.

Run:  python -m housemate.agent --extract
"""
from __future__ import annotations

from housemate.tools import House

EXTRACT_REQUEST = """Read the entire house chat with read_chat (keep paging until next_offset is null).
Then:
1. For every real agreement, call record_decision once. A real agreement is one that someone
   confirms or the group clearly accepts. Skip suggestions nobody agreed to.
2. For every concrete job with a person responsible, call add_task with the assignee,
   any due day or date, and the source message ids.
Finally, reply with a short summary: decisions first, then tasks, each with its [msg-...] citations."""

# Answer key. Each item lists the message ids that support it; citing any one counts as found.
GOLDEN_DECISIONS = {
    "bathroom rotation": {"msg-003", "msg-004"},
    "laundry slots": {"msg-006", "msg-007"},
    "costco runs": {"msg-009", "msg-010", "msg-011"},
    "sunday dinner": {"msg-012", "msg-013", "msg-014"},
}
GOLDEN_TASKS = {
    "trash and recycling (Fuchu)": {"msg-002"},
    "first costco run (Anish)": {"msg-010", "msg-011"},
    "first sunday dinner (Shreeya and Fuchu)": {"msg-014"},
    # Added after the first real run: the agent found this and the original key had missed it.
    "put bathroom rotation in House Tasks (Shreeya)": {"msg-004"},
}


def _recall(golden: dict, records: list) -> tuple[float, list]:
    cited = [set(r.get("source_ids", [])) for r in records]
    missed = [name for name, ids in golden.items() if not any(ids & c for c in cited)]
    return (len(golden) - len(missed)) / len(golden), missed


def score_extraction(house: House) -> dict:
    decision_recall, missed_decisions = _recall(GOLDEN_DECISIONS, house.decisions)
    task_recall, missed_tasks = _recall(GOLDEN_TASKS, house.tasks)
    # Decisions that match nothing in the key: not always wrong, but worth a human look.
    golden_ids = set().union(*GOLDEN_DECISIONS.values())
    extra = [d["summary"] for d in house.decisions if not set(d["source_ids"]) & golden_ids]
    return {
        "decision_recall": decision_recall, "missed_decisions": missed_decisions,
        "task_recall": task_recall, "missed_tasks": missed_tasks,
        "extra_decisions_to_review": extra,
    }
