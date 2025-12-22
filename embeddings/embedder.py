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
DEFAULT_CHUNK_SIZE = 80
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_MAX_CHUNK_CHARS = 512
DEFAULT_MAX_TOKEN_CHARS = 100
DEFAULT_EMBED_BATCH_SIZE = 1
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
    return content.lstrip("\ufeff").strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_token_chars: int = DEFAULT_MAX_TOKEN_CHARS,
) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)
    if max_chars <= 0:
        msg = "max_chars must be a positive integer"
        raise ValueError(msg)
    if max_token_chars <= 0:
        msg = "max_token_chars must be a positive integer"
        raise ValueError(msg)
    if max_token_chars > max_chars:
        msg = "max_token_chars must be <= max_chars"
        raise ValueError(msg)

    raw_tokens = text.split()
    if not raw_tokens:
        return []

    # Split very long tokens to avoid exceeding embedding context limits.
    tokens: List[str] = []
    for token in raw_tokens:
        if len(token) <= max_token_chars:
            tokens.append(token)
        else:
            tokens.extend(token[i:i + max_token_chars] for i in range(0, len(token), max_token_chars))
    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        window = tokens[start:end]
        current: List[str] = []
        current_len = 0

        for token in window:
            if len(token) > max_chars:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                for i in range(0, len(token), max_chars):
                    chunks.append(token[i:i + max_chars])
                continue

            add_len = len(token) + (1 if current else 0)
            if current and current_len + add_len > max_chars:
                chunks.append(" ".join(current))
                current = [token]
                current_len = len(token)
            else:
                current.append(token)
                current_len += add_len

        if current:
            chunks.append(" ".join(current))

        start += step

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
    inputs = list(texts)
    if not inputs:
        raise ValueError("Embedding için boş girdi gönderilemez.")

    def _error_detail(resp: requests.Response) -> str:
        try:
            payload = resp.json()
        except ValueError:
            return resp.text.strip() or f"HTTP {resp.status_code}"
        return payload.get("error") or payload.get("message") or f"HTTP {resp.status_code}"

    def _normalize_embeddings(raw_embeddings: object) -> List[List[float]]:
        if not isinstance(raw_embeddings, list) or not raw_embeddings:
            raise ValueError("Ollama embed yanıtı beklenen formatta değil.")
        if isinstance(raw_embeddings[0], (int, float)):
            return [raw_embeddings]  # tek giriş için düz liste dönen eski sürümler
        return raw_embeddings  # type: ignore[return-value]

    def _request_embed(payload: dict) -> List[List[float]]:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        raw_embeddings = response.json().get("embeddings")
        if raw_embeddings is None:
            raise ValueError("Ollama embed yanıtı 'embeddings' alanını içermiyor.")
        return _normalize_embeddings(raw_embeddings)

    def _embed_batch(batch: List[str]) -> np.ndarray:
        # Primary attempt: batch input with /api/embed
        try:
            payload = {"model": model, "input": batch}
            return np.array(_request_embed(payload), dtype="float32")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {400, 404}:
                raise

        # Fallback: per-text /api/embed (for servers rejecting batch lists)
        try:
            per_text_embeddings = []
            for text in batch:
                raw = _request_embed({"model": model, "input": text})
                per_text_embeddings.append(raw[0])
            return np.array(per_text_embeddings, dtype="float32")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {400, 404}:
                raise

        # Legacy fallback: /api/embeddings per text
        legacy_embeddings = []
        for text in batch:
            legacy_payload = {"model": model, "prompt": text}
            legacy_response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json=legacy_payload,
                timeout=60,
            )
            if not legacy_response.ok:
                detail = _error_detail(legacy_response)
                raise ValueError(
                    f"Ollama embeddings istegi basarisiz: {detail}. "
                    f"Modelin indirildiginden emin olun (ollama pull {model})."
                )
            legacy = legacy_response.json().get("embedding")
            if legacy is None:
                raise ValueError("Ollama embeddings yanıtı 'embedding' alanını içermiyor.")
            legacy_embeddings.append(legacy)
        return np.array(legacy_embeddings, dtype="float32")

    all_embeddings = []
    batch_size = max(1, DEFAULT_EMBED_BATCH_SIZE)
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        all_embeddings.append(_embed_batch(batch))

    return np.vstack(all_embeddings)


def embed_document(path: Path, model: str = DEFAULT_EMBED_MODEL) -> tuple[np.ndarray, List[DocumentChunk]]:
    """Return embeddings and chunk metadata for a given document path."""
    chunks = list(iter_document_chunks(path))
    if not chunks:
        raise ValueError(f"Belge boş veya yalnızca boşluklardan oluşuyor: {path}")
    vectors = embed_texts((chunk.text for chunk in chunks), model=model)
    return vectors, chunks
