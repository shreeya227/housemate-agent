# HouseMate Agent

An agentic RAG assistant for a shared house of 7 housemates. It reads the house group chat, answers questions with citations to the exact messages, pulls out decisions and tasks, and plans meals from a recipe collection against what is in the pantry.

Built in Python on the Claude API (tool use), with keyword retrieval, a grounding check on every citation, and evaluations for both retrieval and extraction.

## What it does

**Answer questions with citations**

```
$ python -m housemate.agent "What did we decide about laundry on weekends?"
[step 1] search_chat({"query": "laundry weekend", "k": 5})

On weekends, you book a 2-hour laundry slot in the chat [msg-007]. Rohan said the same thing [msg-006].
Weekdays after 9pm are open to anyone [msg-007].
```

**Extract decisions and tasks from the chat**

```
$ python -m housemate.agent --extract
[step 1] read_chat({})
[step 2] record_decision({"topic": "bathrooms", "summary": "Bathrooms rotate weekly, two people per week.", "source_ids": ["msg-003", "msg-004"]})
...
[step 8] add_task({"title": "Drive first Costco run", "assignee": "Anish", "due": "2026-10-12", "source_ids": ["msg-010", "msg-011"]})

Things to check:
- Grocery scope: Priya's proposal got no reply, so I didn't record it as a decision. [msg-008]
- Trash: Fuchu volunteered, but nobody confirmed it. I added it as a task, not a decision. [msg-002]

Decision recall: 100%
Task recall:     100%
```

**Plan meals from the pantry**

```
$ python -m housemate.agent "Plan a vegetarian Sunday dinner for the house and add what we're missing to the grocery list"
[step 1] search_chat({"query": "Sunday dinner", "k": 5})
[step 2] search_chat({"query": "dietary allergy vegetarian vegan gluten nut", "k": 5})
...
[step 8] missing_ingredients({"recipe_id": "r1", "people": 7})

Sunday dinner: dal tadka with rice. It must be vegetarian (Sita and Priya don't eat meat) [msg-013].
Two batches feed 7, and the pantry already has every ingredient, so nothing was added to the grocery list.
```

## How it works

```
question ──> Claude ──(tool calls)──> tools ──> results with message ids ──> Claude ──> cited answer
               ^                                                                │
               └──────────────── loop until done (max 12 rounds) ──────────────┘
                                                                                ▼
                                                     grounding check: every [msg-id] cited
                                                     must have been returned by a tool
```

| Module | Role |
|---|---|
| `housemate/ingest.py` | Parses the chat, recipes, and pantry. Every message gets a stable id (`msg-007`) so answers can be cited and checked. |
| `housemate/search.py` | BM25 keyword search over messages and recipes, written from scratch. |
| `housemate/tools.py` | 8 tools: `search_chat`, `read_chat`, `search_recipes`, `check_pantry`, `missing_ingredients`, `add_grocery_item`, `record_decision`, `add_task`. Records that cite message ids that don't exist are rejected. Tool errors come back as data so the agent can recover. |
| `housemate/agent.py` | The tool-use loop: step limit, stop-reason handling, and the grounding check. |
| `housemate/evaluate.py` | Retrieval eval (Recall@3) on a labeled question set. |
| `housemate/extract.py` | Extraction request and recall scoring against a hand-labeled answer key. |

## Evaluation

| Eval | Result | Notes |
|---|---|---|
| Retrieval, Recall@3 (11 questions) | **91%** | Misses synonym questions ("rubbish" vs "trash"). Keyword search matches words, not meaning. |
| Decision extraction recall | **100%** | 4 of 4 labeled decisions, with correct citations |
| Task extraction recall | **100%** | All labeled tasks. The agent also found a task the original answer key had missed, so the key was updated. |

The datasets are small and hand-built. The point is to measure behavior and catch regressions, not to claim benchmark numbers.

## A bug the evals caught

The first extraction run scored **0%**: the agent read the chat, then returned a blank answer and saved nothing. Claude's response was being cut off at the 1,024-token output limit while it was writing several tool calls, and the loop treated any non-`tool_use` stop as a finished answer. The fix was to raise the limit to 4,096 tokens and make any unexpected stop reason (such as `max_tokens`) fail loudly. A regression test now covers it. Without the eval, the agent would have looked like it worked while saving nothing.

## Run it

Requires Python 3.9+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 28 tests, no API key needed (uses a fake model client)
python -m housemate.evaluate        # retrieval Recall@3
```

To run the live agent, add your key to a `.env` file (it is git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python -m housemate.agent "Who drives the first Costco run?"
python -m housemate.agent --extract
```

## Roadmap

- Semantic (vector) search alongside BM25, measured with the same Recall@3 eval, to fix the synonym misses
- Ingredient quantities, so the agent knows whether 2 cans of chickpeas are enough
- Multi-turn conversations, so follow-ups like "yes, add them" work
- FastAPI service with PostgreSQL and Redis, connected to the House Tasks app
