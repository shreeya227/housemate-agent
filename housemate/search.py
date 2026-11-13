"""Step 2: keyword search with citations (the "R" in RAG).

We use BM25, the classic ranking formula behind search engines like
Elasticsearch. The idea in plain words:
  - A document scores higher when it contains the query's words.
  - Rare words count more than common ones ("costco" beats "the").
  - Repeating a word helps, but with diminishing returns.
  - Long documents don't win just because they have more words.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "let", "lets", "me", "my", "of",
    "on", "or", "our", "so", "the", "this", "to", "us", "we", "what", "when",
    "where", "which", "who", "will", "with", "you",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop filler words, and trim plurals."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]  # "bathrooms" -> "bathroom", "weekends" -> "weekend"
        tokens.append(word)
    return tokens


@dataclass
class Document:
    id: str        # the citation address, e.g. "msg-007" or "r3"
    text: str      # what we search over
    source: object  # the original Message or Recipe


@dataclass
class Hit:
    doc: Document
    score: float


class BM25Index:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1  # how much repeated words help
        self.b = b    # how much we penalize long documents
        self.tokens = [tokenize(d.text) for d in docs]
        self.avg_len = sum(len(t) for t in self.tokens) / max(len(docs), 1)
        # Document frequency: in how many documents does each word appear?
        df = Counter(word for toks in self.tokens for word in set(toks))
        n = len(docs)
        # Rare words get a high IDF (inverse document frequency) weight.
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        query_words = tokenize(query)
        hits = []
        for doc, toks in zip(self.docs, self.tokens):
            counts = Counter(toks)
            score = 0.0
            for word in query_words:
                if word not in counts:
                    continue
                tf = counts[word]
                length_norm = 1 - self.b + self.b * len(toks) / self.avg_len
                score += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
            if score > 0:
                hits.append(Hit(doc, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def message_docs(messages) -> list[Document]:
    return [Document(m.id, f"{m.author}: {m.text}", m) for m in messages]


def recipe_docs(recipes) -> list[Document]:
    docs = []
    for r in recipes:
        diet = "vegetarian" if r.vegetarian else "meat"
        text = f"{r.name}. {diet}. Tags: {' '.join(r.tags)}. Ingredients: {', '.join(r.ingredients)}"
        docs.append(Document(r.id, text, r))
    return docs
