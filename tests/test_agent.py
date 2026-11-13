"""Agent tests with a fake Claude, so they run offline and for free.

FakeClient replays scripted responses in the same shape as the real API,
and records what the agent sent, so we can check the loop's behavior.
"""
from types import SimpleNamespace

from housemate.agent import run_agent
from housemate.tools import House


def text(t):
    return {"type": "text", "text": t}


def tool_call(id_, name, input_):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.sent = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **kwargs):
        # Snapshot the conversation as it was at this call (the agent keeps appending to it).
        self.sent.append({**kwargs, "messages": list(kwargs["messages"])})
        stop_reason, content = self.scripted.pop(0)
        return SimpleNamespace(stop_reason=stop_reason, content=content)


def test_agent_searches_then_answers_with_verified_citation():
    fake = FakeClient([
        ("tool_use", [tool_call("t1", "search_chat", {"query": "laundry weekend"})]),
        ("end_turn", [text("Weekdays after 9pm are open; on weekends book a 2 hour slot [msg-007].")]),
    ])
    result = run_agent("What did we decide about laundry on weekends?", fake)
    assert result.steps[0]["tool"] == "search_chat"
    assert result.cited_ids == ["msg-007"]
    assert result.unverified_ids == []


def test_tool_results_are_sent_back_to_claude():
    fake = FakeClient([
        ("tool_use", [tool_call("t1", "check_pantry", {})]),
        ("end_turn", [text("You have rice.")]),
    ])
    run_agent("What's in the pantry?", fake)
    last_user_turn = fake.sent[1]["messages"][-1]
    assert last_user_turn["content"][0]["tool_use_id"] == "t1"
    assert "rice" in last_user_turn["content"][0]["content"]


def test_made_up_citation_is_flagged():
    fake = FakeClient([("end_turn", [text("Rohan cleans the fridge [msg-012].")])])
    result = run_agent("Who cleans the fridge?", fake)
    assert result.unverified_ids == ["msg-012"]  # cited without ever searching


def test_meal_planning_adds_only_missing_items():
    house = House.from_data()
    fake = FakeClient([
        ("tool_use", [tool_call("t1", "missing_ingredients", {"recipe_id": "r5"})]),
        ("tool_use", [tool_call("t2", "add_grocery_item", {"item": "ginger", "reason": "chana masala"}),
                      tool_call("t3", "add_grocery_item", {"item": "garam masala", "reason": "chana masala"})]),
        ("end_turn", [text("Added ginger and garam masala for chana masala.")]),
    ])
    run_agent("Plan Sunday dinner", fake, house)
    assert [g["item"] for g in house.grocery_list] == ["ginger", "garam masala"]


def test_tool_errors_are_reported_back_not_crashed():
    fake = FakeClient([
        ("tool_use", [tool_call("t1", "missing_ingredients", {"recipe_id": "r99"})]),
        ("end_turn", [text("I couldn't find that recipe.")]),
    ])
    result = run_agent("What's missing for r99?", fake)
    tool_result = fake.sent[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert result.answer == "I couldn't find that recipe."


def test_step_limit_stops_endless_loops():
    looping = [("tool_use", [tool_call(f"t{i}", "check_pantry", {})]) for i in range(10)]
    result = run_agent("loop forever", FakeClient(looping), max_steps=3)
    assert result.answer.startswith("Stopped")
    assert len(result.steps) == 3
