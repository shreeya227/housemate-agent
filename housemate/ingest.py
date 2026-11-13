"""Step 1: turn raw house data into clean, citable records.

Every chat message gets a stable ID (msg-001, msg-002, ...) so that later,
when the agent answers a question, it can point back to the exact message.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Matches lines like: [2026-10-02 19:04] Shreeya: Hi all!
LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] ([^:]+): (.+)$")


@dataclass
class Message:
    id: str
    sent_at: datetime
    author: str
    text: str

    def citation(self) -> str:
        return f"{self.id} ({self.author}, {self.sent_at:%b %d %H:%M})"


@dataclass
class Recipe:
    id: str
    name: str
    serves: int
    vegetarian: bool
    ingredients: dict
    tags: list


def load_chat(path: str | Path) -> list[Message]:
    messages = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = LINE.match(line)
        if match:
            sent_at, author, text = match.groups()
            messages.append(Message(
                id=f"msg-{len(messages) + 1:03d}",
                sent_at=datetime.strptime(sent_at, "%Y-%m-%d %H:%M"),
                author=author.strip(),
                text=text.strip(),
            ))
        elif messages:
            # A line without a timestamp continues the previous message.
            messages[-1].text += " " + line
    return messages


def load_recipes(path: str | Path) -> list[Recipe]:
    return [Recipe(**r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]


def load_pantry(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
