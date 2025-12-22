"""Terminal tabanlı Gemma 3 4B RAG demosu."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import List

import numpy as np

from embeddings.embedder import (
    DEFAULT_EMBED_MODEL,
    DocumentChunk,
    embed_document,
    embed_texts,
)
from generator.gemma import MISSING_ANSWER, build_prompt, generate_with_gemma
from retriever import INDEX_VERSION
from retriever.faiss_store import FaissStore

INDEX_PATH = Path("embeddings/index.faiss")
METADATA_PATH = Path("embeddings/chunks.json")
DEFAULT_DATA_PATH = Path("data/documents.txt")
TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def chunks_to_dicts(chunks: List[DocumentChunk], doc_hash: str, embed_model: str) -> List[dict]:
    """Convert DocumentChunk instances into plain dicts for persistence."""
    return [
        {
            "text": chunk.text,
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "doc_hash": doc_hash,
            "index_version": INDEX_VERSION,
            "embed_model": embed_model,
        }
        for chunk in chunks
    ]


def ensure_index(document_path: Path, embed_model: str = DEFAULT_EMBED_MODEL) -> FaissStore:
    """Create a FAISS index from the document if it does not exist yet."""
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        try:
            store = FaissStore.load(INDEX_PATH, METADATA_PATH)
            # If stored metadata belongs to a different document or content changed, rebuild
            sources = {m.get("source") for m in store.metadata}
            current_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
            hashes = {m.get("doc_hash") for m in store.metadata}
            versions = {m.get("index_version") for m in store.metadata}
            models = {m.get("embed_model") for m in store.metadata}
            if (
                sources != {str(document_path)}
                or hashes != {current_hash}
                or versions != {INDEX_VERSION}
                or models != {embed_model}
                or store.index.ntotal != len(store.metadata)
            ):
                raise ValueError("mismatch")
            return store
        except Exception:
            pass

    vectors, chunks = embed_document(document_path, model=embed_model)
    store = FaissStore(dimension=vectors.shape[1])
    doc_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
    store.add(vectors, chunks_to_dicts(chunks, doc_hash, embed_model))
    store.save(INDEX_PATH, METADATA_PATH)
    return store


def _keyword_overlap(question: str, text: str) -> int:
    q_tokens = set(TOKEN_PATTERN.findall(question.lower()))
    if not q_tokens:
        return 0
    text_tokens = set(TOKEN_PATTERN.findall(text.lower()))
    return len(q_tokens & text_tokens)


def retrieve_context(
    store: FaissStore,
    question: str,
    *,
    k: int = 20,
    top_k: int = 3,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[str]:
    """Embed the question, retrieve similar chunks, and rerank by overlap."""
    query_vector = embed_texts([question], model=embed_model)[0]
    query_vector = np.expand_dims(query_vector, axis=0)
    results = store.search(query_vector, k=k)
    candidates: List[tuple[int, int, float, dict]] = []
    seen_keys = set()

    for meta, score in results:
        key = (meta.get("source"), meta.get("chunk_id"))
        seen_keys.add(key)
        overlap = _keyword_overlap(question, meta["text"])
        candidates.append((1 if overlap else 0, overlap, float(score), meta))

    # Lexical fallback: add best overlap matches from entire corpus
    lexical_matches = []
    for meta in store.metadata:
        key = (meta.get("source"), meta.get("chunk_id"))
        if key in seen_keys:
            continue
        overlap = _keyword_overlap(question, meta["text"])
        if overlap:
            lexical_matches.append((overlap, meta))

    lexical_matches.sort(key=lambda item: item[0], reverse=True)
    for overlap, meta in lexical_matches[: max(0, top_k - len(candidates)) + top_k]:
        key = (meta.get("source"), meta.get("chunk_id"))
        if key in seen_keys:
            continue
        candidates.append((1, overlap, 0.0, meta))
        seen_keys.add(key)

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [meta["text"] for _, _, _, meta in candidates[:top_k]]


def interactive_session(store: FaissStore, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
    """Simple REPL loop for asking questions to the RAG chatbot."""
    print("Gemma 3 4B RAG demo hazır. Çıkmak için Ctrl+C veya boş satır bırakın.\n")

    while True:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşmek üzere!")
            break

        if not question:
            print("Boş soru algılandı. Çıkılıyor.")
            break

        context_chunks = retrieve_context(store, question, embed_model=embed_model)
        if not context_chunks or all(_keyword_overlap(question, chunk) == 0 for chunk in context_chunks):
            answer = MISSING_ANSWER
        else:
            prompt = build_prompt(question, context_chunks)
            answer = generate_with_gemma(prompt)
        print(f"\nCevap:\n{answer}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 3 4B tabanlı basit RAG sohbet botu")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="RAG için kullanılacak düz metin dosyası",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="Ollama embedding modeli (varsayılan: all-minilm)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Doküman bulunamadı: {args.data}")

    store = ensure_index(args.data, embed_model=args.embed_model)
    interactive_session(store, embed_model=args.embed_model)


if __name__ == "__main__":
    main()
