from pathlib import Path

from housemate.ingest import load_chat, load_pantry, load_recipes

DATA = Path(__file__).parent.parent / "data"


def test_every_message_gets_a_stable_id_and_author():
    messages = load_chat(DATA / "house_chat.txt")
    assert len(messages) == 15
    assert messages[0].id == "msg-001"
    assert messages[1].author == "Fuchu"
    assert "Tuesday" in messages[1].text


def test_citation_points_back_to_the_source():
    first = load_chat(DATA / "house_chat.txt")[0]
    assert first.citation() == "msg-001 (Shreeya, Oct 02 19:04)"


def test_line_without_timestamp_joins_previous_message(tmp_path):
    chat = tmp_path / "chat.txt"
    chat.write_text("[2026-10-02 19:04] Sita: Groceries list:\nmilk and eggs\n")
    messages = load_chat(chat)
    assert len(messages) == 1
    assert messages[0].text == "Groceries list: milk and eggs"


def test_recipes_and_pantry_load():
    recipes = load_recipes(DATA / "recipes.json")
    assert {r.name for r in recipes if not r.vegetarian} == {"Chicken momo"}
    assert "rice" in load_pantry(DATA / "pantry.json")
