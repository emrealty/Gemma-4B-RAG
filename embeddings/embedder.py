"""Utilities for turning raw documents into dense embeddings via Ollama."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests

DEFAULT_EMBED_MODEL = "all-minilm"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


@dataclass
class DocumentChunk:
    """Lightweight container for storing chunk metadata."""

    text: str
    source: str
    chunk_id: int


def load_document(path: Path) -> str:
    """Read a UTF-8 text document from disk."""
    content = path.read_text(encoding="utf-8")
    return content.strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)

    tokens = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def iter_document_chunks(path: Path) -> Iterable[DocumentChunk]:
    """Yield document chunks enriched with metadata about their origin."""
    text = load_document(path)
    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):
        yield DocumentChunk(text=chunk, source=str(path), chunk_id=idx)


def embed_texts(
    texts: Iterable[str],
    model: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """Request embeddings from the local Ollama server."""
    payload = {"model": model, "input": list(texts)}
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()

    raw_embeddings = response.json()["embeddings"]
    return np.array(raw_embeddings, dtype="float32")


def embed_document(path: Path, model: str = DEFAULT_EMBED_MODEL) -> tuple[np.ndarray, List[DocumentChunk]]:
    """Return embeddings and chunk metadata for a given document path."""
    chunks = list(iter_document_chunks(path))
    vectors = embed_texts((chunk.text for chunk in chunks), model=model)
    return vectors, chunks
