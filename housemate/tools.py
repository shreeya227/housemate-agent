"""Step 3: tools the agent can call.

Each tool has two parts:
  1. A plain Python method that does the work.
  2. A schema (name, description, input fields) that tells Claude what the
     tool does and how to call it. Claude reads only the schema, never the code.

Tools return plain dicts so the results can be sent back to Claude as JSON.
Search results always carry IDs so the final answer can cite them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from housemate.ingest import load_chat, load_pantry, load_recipes
from housemate.search import BM25Index, message_docs, recipe_docs

DATA = Path(__file__).parent.parent / "data"
HOUSE_SIZE = 7


@dataclass
class House:
    """Everything the agent can read or change."""
    messages: list
    recipes: list
    pantry: dict
    grocery_list: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    decisions: list = field(default_factory=list)

    def __post_init__(self):
        self.chat_index = BM25Index(message_docs(self.messages))
        self.recipe_index = BM25Index(recipe_docs(self.recipes))
        self.recipes_by_id = {r.id: r for r in self.recipes}

    @classmethod
    def from_data(cls, data_dir: Path = DATA) -> "House":
        return cls(
            messages=load_chat(data_dir / "house_chat.txt"),
            recipes=load_recipes(data_dir / "recipes.json"),
            pantry=load_pantry(data_dir / "pantry.json"),
        )

    # ---------- read-only tools ----------

    def search_chat(self, query: str, k: int = 3) -> dict:
        hits = self.chat_index.search(query, k=k)
        return {"results": [
            {"id": h.doc.id, "author": h.doc.source.author,
             "sent_at": h.doc.source.sent_at.strftime("%Y-%m-%d %H:%M"),
             "text": h.doc.source.text, "score": round(h.score, 2)}
            for h in hits
        ]}

    def search_recipes(self, query: str, vegetarian_only: bool = False, k: int = 3) -> dict:
        hits = self.recipe_index.search(query, k=len(self.recipes))
        recipes = [h.doc.source for h in hits]
        if vegetarian_only:
            recipes = [r for r in recipes if r.vegetarian]
        return {"results": [
            {"id": r.id, "name": r.name, "serves": r.serves,
             "vegetarian": r.vegetarian, "ingredients": list(r.ingredients)}
            for r in recipes[:k]
        ]}

    def read_chat(self, offset: int = 0, limit: int = 50) -> dict:
        """Read the chat in order, one page at a time (for tasks that need every message)."""
        page = self.messages[offset:offset + limit]
        next_offset = offset + len(page)
        return {
            "results": [{"id": m.id, "author": m.author,
                         "sent_at": m.sent_at.strftime("%Y-%m-%d %H:%M"), "text": m.text}
                        for m in page],
            "next_offset": next_offset if next_offset < len(self.messages) else None,
        }

    def check_pantry(self) -> dict:
        return {"pantry": dict(self.pantry)}

    def missing_ingredients(self, recipe_id: str, people: int = HOUSE_SIZE) -> dict:
        recipe = self.recipes_by_id.get(recipe_id)
        if recipe is None:
            return {"error": f"No recipe with id '{recipe_id}'"}
        have = [i for i in recipe.ingredients if i in self.pantry]
        missing = [i for i in recipe.ingredients if i not in self.pantry]
        return {
            "recipe_id": recipe.id, "recipe": recipe.name,
            "batches_needed": math.ceil(people / recipe.serves),
            "have": have, "missing": missing,
        }

    # ---------- action tools (they change state) ----------

    def add_grocery_item(self, item: str, reason: str = "") -> dict:
        name = item.strip().lower()
        if any(g["item"] == name for g in self.grocery_list):
            return {"status": "already_on_list", "item": name}
        self.grocery_list.append({"item": name, "reason": reason})
        return {"status": "added", "item": name, "list_size": len(self.grocery_list)}

    def _check_sources(self, source_ids: list) -> str | None:
        """Refuse made-up citations: every record must point to real messages."""
        known = {m.id for m in self.messages}
        unknown = [s for s in source_ids if s not in known]
        return f"Unknown source message ids: {unknown}" if unknown else None

    def record_decision(self, topic: str, summary: str, source_ids: list) -> dict:
        if not source_ids:
            return {"error": "A decision needs at least one source message id"}
        problem = self._check_sources(source_ids)
        if problem:
            return {"error": problem}
        decision = {"id": f"dec-{len(self.decisions) + 1:03d}", "topic": topic,
                    "summary": summary, "source_ids": source_ids}
        self.decisions.append(decision)
        return {"status": "recorded", "decision": decision}

    def add_task(self, title: str, assignee: str, due: str = "", source_ids: list | None = None) -> dict:
        source_ids = source_ids or []
        problem = self._check_sources(source_ids)
        if problem:
            return {"error": problem}
        task = {"id": f"task-{len(self.tasks) + 1:03d}", "title": title,
                "assignee": assignee, "due": due, "source_ids": source_ids}
        self.tasks.append(task)
        return {"status": "added", "task": task}


# What Claude sees. Descriptions matter: they are how Claude decides which tool to use.
TOOL_SCHEMAS = [
    {
        "name": "search_chat",
        "description": "Search the house group chat for messages about a topic. Use this to find what housemates said or decided. Returns messages with ids to cite.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Keywords to search for, e.g. 'laundry weekend'"},
            "k": {"type": "integer", "description": "How many messages to return (default 3)"},
        }, "required": ["query"]},
    },
    {
        "name": "search_recipes",
        "description": "Search the house recipe collection by dish, cuisine, tag, or ingredient.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"},
            "vegetarian_only": {"type": "boolean", "description": "Only return vegetarian recipes"},
            "k": {"type": "integer"},
        }, "required": ["query"]},
    },
    {
        "name": "read_chat",
        "description": "Read the house chat in order, page by page. Use this when you need every message (for example to find all decisions), not just the best matches. Keep calling with next_offset until it is null.",
        "input_schema": {"type": "object", "properties": {
            "offset": {"type": "integer", "description": "Index of the first message to read (default 0)"},
            "limit": {"type": "integer", "description": "Messages per page (default 50)"},
        }},
    },
    {
        "name": "check_pantry",
        "description": "List the ingredients currently in the shared pantry.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "missing_ingredients",
        "description": "For a recipe id, list which ingredients the pantry has and which are missing, and how many batches feed the house.",
        "input_schema": {"type": "object", "properties": {
            "recipe_id": {"type": "string"},
            "people": {"type": "integer", "description": "Number of people to feed (default 7)"},
        }, "required": ["recipe_id"]},
    },
    {
        "name": "add_grocery_item",
        "description": "Add one item to the shared grocery list. Skips duplicates.",
        "input_schema": {"type": "object", "properties": {
            "item": {"type": "string"},
            "reason": {"type": "string", "description": "Why it is needed, e.g. 'for chana masala'"},
        }, "required": ["item"]},
    },
    {
        "name": "record_decision",
        "description": "Record something the house agreed on. Only record real agreements (someone confirms, or the group clearly accepts), not one-off suggestions. Always cite the message ids where it was proposed and agreed.",
        "input_schema": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "Short topic, e.g. 'laundry', 'groceries', 'chores'"},
            "summary": {"type": "string", "description": "One sentence stating the decision"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        }, "required": ["topic", "summary", "source_ids"]},
    },
    {
        "name": "add_task",
        "description": "Create a house task. Always pass the chat message ids that support it in source_ids.",
        "input_schema": {"type": "object", "properties": {
            "title": {"type": "string"},
            "assignee": {"type": "string"},
            "due": {"type": "string", "description": "Due date or day, e.g. '2026-10-12' or 'Sunday'"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        }, "required": ["title", "assignee"]},
    },
]


def run_tool(house: House, name: str, tool_input: dict) -> dict:
    """Call a tool by name. Errors come back as data so the agent can recover."""
    allowed = {schema["name"] for schema in TOOL_SCHEMAS}
    if name not in allowed:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return getattr(house, name)(**tool_input)
    except TypeError as exc:  # wrong or missing arguments
        return {"error": f"Bad input for {name}: {exc}"}
