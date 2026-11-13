from pathlib import Path

from housemate.evaluate import recall_at_k
from housemate.ingest import load_chat, load_recipes
from housemate.search import BM25Index, message_docs, recipe_docs, tokenize

DATA = Path(__file__).parent.parent / "data"


def chat_index():
    return BM25Index(message_docs(load_chat(DATA / "house_chat.txt")))


def test_tokenize_drops_filler_words_and_plurals():
    assert tokenize("What are the Bathrooms rules?") == ["bathroom", "rule"]


def test_search_returns_cited_messages_best_first():
    hits = chat_index().search("laundry on weekends", k=3)
    assert hits[0].doc.id in {"msg-006", "msg-007"}
    assert hits[0].score >= hits[-1].score


def test_rare_words_beat_common_ones():
    # "costco" appears in few messages, so it should decide the ranking.
    top = chat_index().search("costco", k=1)[0]
    assert "Costco" in top.doc.text


def test_no_match_returns_nothing():
    assert chat_index().search("swimming pool") == []


def test_recipe_search_finds_vegetarian_nepali_dishes():
    index = BM25Index(recipe_docs(load_recipes(DATA / "recipes.json")))
    names = [h.doc.source.name for h in index.search("vegetarian nepali", k=2)]
    assert "Chicken momo" not in names


def test_keyword_search_quality_baseline():
    # Keyword search misses synonyms ("rubbish" vs "trash"); Step 6 adds meaning-based search.
    assert recall_at_k(chat_index(), k=3) >= 0.9
