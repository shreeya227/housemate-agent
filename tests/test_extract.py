from housemate.agent import run_agent
from housemate.extract import score_extraction
from housemate.tools import House, run_tool
from tests.test_agent import FakeClient, text, tool_call


def test_read_chat_pages_through_every_message():
    house = House.from_data()
    first = run_tool(house, "read_chat", {"offset": 0, "limit": 10})
    second = run_tool(house, "read_chat", {"offset": first["next_offset"], "limit": 10})
    ids = [m["id"] for m in first["results"] + second["results"]]
    assert ids == [f"msg-{i:03d}" for i in range(1, 16)]
    assert second["next_offset"] is None


def test_decision_needs_real_citations():
    house = House.from_data()
    assert "error" in run_tool(house, "record_decision", {"topic": "x", "summary": "y", "source_ids": []})
    assert "error" in run_tool(house, "record_decision", {"topic": "x", "summary": "y", "source_ids": ["msg-404"]})
    ok = run_tool(house, "record_decision", {"topic": "laundry", "summary": "Weekend slots", "source_ids": ["msg-007"]})
    assert ok["status"] == "recorded"


def test_extraction_flow_and_scoring():
    house = House.from_data()
    fake = FakeClient([
        ("tool_use", [tool_call("t1", "read_chat", {})]),
        ("tool_use", [
            tool_call("t2", "record_decision", {"topic": "chores", "summary": "Bathrooms rotate weekly, two people", "source_ids": ["msg-004"]}),
            tool_call("t3", "record_decision", {"topic": "laundry", "summary": "Weekdays after 9pm open; weekends book 2 hours", "source_ids": ["msg-007"]}),
            tool_call("t4", "record_decision", {"topic": "groceries", "summary": "Costco every two weeks, split 7 ways", "source_ids": ["msg-011"]}),
            tool_call("t5", "add_task", {"title": "First Costco run", "assignee": "Anish", "due": "2026-10-12", "source_ids": ["msg-011"]}),
            tool_call("t6", "add_task", {"title": "Trash out Monday night", "assignee": "Fuchu", "due": "Mondays", "source_ids": ["msg-002"]}),
        ]),
        ("end_turn", [text("Decisions: bathrooms [msg-004], laundry [msg-007], Costco [msg-011].")]),
    ])
    result = run_agent("extract", fake, house)
    assert result.unverified_ids == []

    score = score_extraction(house)
    assert score["decision_recall"] == 0.75           # missed Sunday dinner
    assert score["missed_decisions"] == ["sunday dinner"]
    assert score["task_recall"] == 0.5                # 2 of 4 tasks found
    assert score["extra_decisions_to_review"] == []
