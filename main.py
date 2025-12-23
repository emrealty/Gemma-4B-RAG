"""Terminal tabanlı Gemma 3 4B RAG demosu."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

from embeddings.embedder import (
    DEFAULT_EMBED_MODEL,
    DocumentChunk,
    embed_document,
)
from generator.gemma import MISSING_ANSWER, build_prompt, generate_with_gemma
from retriever import INDEX_VERSION
from retriever.faiss_store import FaissStore
from retriever.query import has_relevant_context, retrieve_context

INDEX_PATH = Path("embeddings/index.faiss")
METADATA_PATH = Path("embeddings/chunks.json")
DEFAULT_DATA_PATH = Path("data/documents.txt")


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
        if not context_chunks or not has_relevant_context(question, context_chunks):
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
