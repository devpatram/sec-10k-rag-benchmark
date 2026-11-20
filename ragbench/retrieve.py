"""The retrievers: dense (embeddings + FAISS), BM25, hybrid (RRF of the two), and query decomposition."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from .corpus import Page
from .metrics import rrf

TOKEN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Retriever:
    name = "BM25"

    def __init__(self, pages: list[Page]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(p.text) for p in pages])

    def rank(self, query: str, n: int = 50) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores, kind="stable")[:n]]


class DenseRetriever:
    name = "Dense"
    # bge-v1.5 models expect this instruction on the query only, never on the documents.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, pages: list[Page], model: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        vectors = self.model.encode([p.text for p in pages], batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of unit vectors = cosine similarity
        self.index.add(vectors)
        self.dimension = int(vectors.shape[1])

    def rank(self, query: str, n: int = 50) -> list[int]:
        q = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        _, ids = self.index.search(q, min(n, self.index.ntotal))
        return [int(i) for i in ids[0]]


class HybridRetriever:
    name = "Hybrid RRF"

    def __init__(self, dense, sparse, c: int = 60):
        self.dense, self.sparse, self.c = dense, sparse, c

    def rank(self, query: str, n: int = 50) -> list[int]:
        return rrf([self.dense.rank(query, n), self.sparse.rank(query, n)], self.c)[:n]


class DecompositionRetriever:
    """Splits the question into sub-questions, retrieves for each with a base retriever, and pools them with RRF."""

    def __init__(self, base, split: Callable[[str], list[str]], c: int = 60):
        self.base, self.split, self.c = base, split, c
        self.name = f"Decomposition + {base.name}"
        self.last_subquestions: list[str] = []

    def rank(self, query: str, n: int = 50) -> list[int]:
        self.last_subquestions = self.split(query) or [query]
        return rrf([self.base.rank(q, n) for q in self.last_subquestions], self.c)[:n]


def parse_subquestions(raw: str, limit: int = 4) -> list[str]:
    """One sub-question per line; list markers stripped; very short lines dropped."""
    out: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        line = re.sub(r"^(?:sub-?)?(?:question|query)\s*\d*\s*:\s*", "", line, flags=re.I).strip()
        if len(line) > 5 and line.lower() not in (o.lower() for o in out):
            out.append(line)
    return out[:limit]
