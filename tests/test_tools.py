from housemate.tools import TOOL_SCHEMAS, House, run_tool


def test_search_chat_returns_citable_messages():
    result = run_tool(House.from_data(), "search_chat", {"query": "laundry weekend"})
    top = result["results"][0]
    assert top["id"] in {"msg-006", "msg-007"}
    assert {"id", "author", "sent_at", "text"} <= set(top)


def test_vegetarian_filter_excludes_meat():
    result = run_tool(House.from_data(), "search_recipes", {"query": "nepali", "vegetarian_only": True})
    assert result["results"]
    assert all(r["vegetarian"] for r in result["results"])


def test_missing_ingredients_compares_recipe_to_pantry():
    result = run_tool(House.from_data(), "missing_ingredients", {"recipe_id": "r5"})  # chana masala
    assert "ginger" in result["missing"]
    assert "chickpeas" in result["have"]
    assert result["batches_needed"] == 2  # serves 6, house of 7


def test_grocery_list_skips_duplicates():
    house = House.from_data()
    run_tool(house, "add_grocery_item", {"item": "Ginger", "reason": "chana masala"})
    again = run_tool(house, "add_grocery_item", {"item": "ginger"})
    assert again["status"] == "already_on_list"
    assert len(house.grocery_list) == 1


def test_task_must_cite_real_messages():
    house = House.from_data()
    ok = run_tool(house, "add_task", {"title": "Costco run", "assignee": "Anish",
                                      "due": "2026-10-12", "source_ids": ["msg-010", "msg-011"]})
    assert ok["status"] == "added"
    bad = run_tool(house, "add_task", {"title": "x", "assignee": "y", "source_ids": ["msg-999"]})
    assert "error" in bad


def test_errors_come_back_as_data_not_crashes():
    house = House.from_data()
    assert "error" in run_tool(house, "delete_everything", {})
    assert "error" in run_tool(house, "missing_ingredients", {"recipe_id": "r99"})
    assert "error" in run_tool(house, "search_chat", {})  # missing query


def test_every_schema_matches_a_method():
    house = House.from_data()
    for schema in TOOL_SCHEMAS:
        assert callable(getattr(house, schema["name"]))
