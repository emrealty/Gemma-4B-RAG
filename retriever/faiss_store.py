"""Simple FAISS-backed retriever for local RAG experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss  # type: ignore
import numpy as np


@dataclass
class FaissStore:
    """Wrap a FAISS index together with chunk metadata."""

    dimension: int
    index: faiss.IndexFlatIP = field(init=False)
    metadata: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Cosine similarity via inner product on L2-normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)

    def add(self, vectors: np.ndarray, chunks: Sequence[dict]) -> None:
        if vectors.shape[1] != self.dimension:
            msg = "Vector dimension mismatch"
            raise ValueError(msg)
        vecs = vectors.astype("float32", copy=False)
        # L2 normalize rows
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self.index.add(vecs)
        self.metadata.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[dict, float]]:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        q = query_vector.astype("float32", copy=False)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q = q / q_norm

        distances, indices = self.index.search(q, k)
        results: List[Tuple[dict, float]] = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk_info = self.metadata[idx]
            results.append((chunk_info, float(dist)))

        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        store = cls(dimension=index.d)
        store.index = index
        store.metadata = metadata
        return store
