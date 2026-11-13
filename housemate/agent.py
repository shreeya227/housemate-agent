"""Step 4: the agent loop.

How an agent works, in four moves that repeat:
  1. Send Claude the conversation plus the tool menu (TOOL_SCHEMAS).
  2. Claude either answers, or asks to call one or more tools.
  3. We run those tools and send the results back.
  4. Repeat until Claude answers (or we hit a step limit, so it can never loop forever).

After the answer, we check grounding: every message id Claude cites must be one
that a tool actually returned during this run. Anything else is flagged.

Run:  python -m housemate.agent "What did we decide about laundry on weekends?"
      python -m housemate.agent --extract
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from housemate.tools import TOOL_SCHEMAS, House, run_tool

MODEL = os.environ.get("HOUSEMATE_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 12

SYSTEM_PROMPT = """You are HouseMate, an assistant for a shared house of 7 housemates.
Use the tools to look things up. Never answer house questions from memory.
When you state something housemates said or decided, cite the message ids in
square brackets, like [msg-007]. Only cite ids that a tool returned.
If the chat does not answer the question, say so plainly.
For meal planning: search recipes, check missing ingredients, and add only the
missing items to the grocery list. Respect dietary needs stated in the chat.
Keep answers short and practical."""

CITATION = re.compile(r"\[(msg-\d{3})\]")


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)       # every tool call, for debugging and demos
    cited_ids: list = field(default_factory=list)   # ids cited in the answer
    unverified_ids: list = field(default_factory=list)  # cited but never returned by a tool


def _block_get(block, key):
    """Read a field from an SDK content block or a plain dict (used by tests)."""
    return block.get(key) if isinstance(block, dict) else getattr(block, key)


def run_agent(question: str, client, house: House | None = None,
              model: str = MODEL, max_steps: int = MAX_STEPS) -> AgentResult:
    house = house or House.from_data()
    messages = [{"role": "user", "content": question}]
    steps = []
    seen_ids = set()  # message ids that tools actually returned

    for _ in range(max_steps):
        response = client.messages.create(
            model=model, max_tokens=1024, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(_block_get(b, "text") for b in response.content
                             if _block_get(b, "type") == "text").strip()
            cited = list(dict.fromkeys(CITATION.findall(answer)))
            return AgentResult(answer, steps, cited, [c for c in cited if c not in seen_ids])

        tool_results = []
        for block in response.content:
            if _block_get(block, "type") != "tool_use":
                continue
            name, tool_input = _block_get(block, "name"), _block_get(block, "input")
            result = run_tool(house, name, tool_input)
            seen_ids.update(r["id"] for r in result.get("results", []) if r.get("id", "").startswith("msg-"))
            steps.append({"tool": name, "input": tool_input, "result": result})
            tool_results.append({
                "type": "tool_result", "tool_use_id": _block_get(block, "id"),
                "content": json.dumps(result), "is_error": "error" in result,
            })
        messages.append({"role": "user", "content": tool_results})

    return AgentResult("Stopped: reached the step limit before finishing.", steps)


def _load_env(path: Path = Path(".env")) -> None:
    """Read KEY=value lines from .env into the environment (no extra package needed)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def main() -> None:
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY. Add it to a .env file in this folder.")
    import anthropic  # imported here so tests run without the package or a key

    from housemate.extract import EXTRACT_REQUEST, score_extraction

    args = sys.argv[1:]
    extract = args[:1] == ["--extract"]
    question = EXTRACT_REQUEST if extract else (" ".join(args) or "What did we decide about laundry on weekends?")
    house = House.from_data()
    result = run_agent(question, anthropic.Anthropic(), house)

    for i, step in enumerate(result.steps, 1):
        print(f"[step {i}] {step['tool']}({json.dumps(step['input'])})")
    print("\n" + result.answer)
    if result.unverified_ids:
        print(f"\nWARNING: cited ids no tool returned: {result.unverified_ids}")
    if house.grocery_list:
        print("\nGrocery list:", ", ".join(g["item"] for g in house.grocery_list))
    if house.decisions:
        print("\nDecisions recorded:", *[f"[{d['topic']}] {d['summary']} {d['source_ids']}" for d in house.decisions], sep="\n  ")
    if house.tasks:
        print("\nTasks:", *[f"{t['title']} -> {t['assignee']} {t['due']} {t['source_ids']}" for t in house.tasks], sep="\n  ")
    if extract:
        score = score_extraction(house)
        print(f"\nDecision recall: {score['decision_recall']:.0%}  missed: {score['missed_decisions']}")
        print(f"Task recall:     {score['task_recall']:.0%}  missed: {score['missed_tasks']}")
        if score["extra_decisions_to_review"]:
            print("Extra decisions to review:", score["extra_decisions_to_review"])


if __name__ == "__main__":
    main()
