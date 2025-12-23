"""Shared query helpers for retrieving relevant chunks from a FAISS store."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import numpy as np

from embeddings.embedder import DEFAULT_EMBED_MODEL, embed_texts
from .faiss_store import FaissStore

TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def keyword_overlap(question: str, text: str) -> int:
    """Return the count of shared keyword tokens between question and text."""
    q_tokens = set(TOKEN_PATTERN.findall(question.lower()))
    if not q_tokens:
        return 0
    text_tokens = set(TOKEN_PATTERN.findall(text.lower()))
    return len(q_tokens & text_tokens)


def has_relevant_context(question: str, contexts: Iterable[str]) -> bool:
    """Check if any context chunk shares at least one keyword with the question."""
    return any(keyword_overlap(question, text) > 0 for text in contexts)


def retrieve_context(
    store: FaissStore,
    question: str,
    *,
    k: int = 20,
    top_k: int = 3,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[str]:
    """Embed the question, retrieve similar chunks, and rerank by keyword overlap."""
    query_vector = embed_texts([question], model=embed_model)[0]
    query_vector = np.expand_dims(query_vector, axis=0)
    results = store.search(query_vector, k=k)

    candidates: List[Tuple[int, int, float, dict]] = []
    seen_keys = set()

    def _add_candidate(meta: dict, score: float, overlap: int | None = None) -> None:
        key = (meta.get("source"), meta.get("chunk_id"))
        if key in seen_keys:
            return
        seen_keys.add(key)
        overlap_score = keyword_overlap(question, meta["text"]) if overlap is None else overlap
        candidates.append((1 if overlap_score else 0, overlap_score, score, meta))

    for meta, score in results:
        _add_candidate(meta, float(score))

    # Lexical fallback: add highest-overlap chunks outside initial vector hits
    if len(candidates) < top_k:
        lexical_matches = []
        for meta in store.metadata:
            key = (meta.get("source"), meta.get("chunk_id"))
            if key in seen_keys:
                continue
            overlap = keyword_overlap(question, meta["text"])
            if overlap:
                lexical_matches.append((overlap, meta))

        lexical_matches.sort(key=lambda item: item[0], reverse=True)
        limit = max(0, top_k - len(candidates)) + top_k
        for overlap, meta in lexical_matches[:limit]:
            _add_candidate(meta, 0.0, overlap)

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [meta["text"] for _, _, _, meta in candidates[:top_k]]
