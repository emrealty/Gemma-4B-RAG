"""Terminal tabanlı Gemma 3 4B RAG demosu."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from embeddings.embedder import (
    DEFAULT_EMBED_MODEL,
    DocumentChunk,
    embed_document,
    embed_texts,
)
from generator.gemma import build_prompt, generate_with_gemma
from retriever.faiss_store import FaissStore

INDEX_PATH = Path("embeddings/index.faiss")
METADATA_PATH = Path("embeddings/chunks.json")
DEFAULT_DATA_PATH = Path("data/documents.txt")


def chunks_to_dicts(chunks: List[DocumentChunk]) -> List[dict]:
    """Convert DocumentChunk instances into plain dicts for persistence."""
    return [
        {
            "text": chunk.text,
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
    ]


def ensure_index(document_path: Path, embed_model: str = DEFAULT_EMBED_MODEL) -> FaissStore:
    """Create a FAISS index from the document if it does not exist yet."""
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        store = FaissStore.load(INDEX_PATH, METADATA_PATH)
        return store

    vectors, chunks = embed_document(document_path, model=embed_model)
    store = FaissStore(dimension=vectors.shape[1])
    store.add(vectors, chunks_to_dicts(chunks))
    store.save(INDEX_PATH, METADATA_PATH)
    return store


def retrieve_context(store: FaissStore, question: str) -> List[str]:
    """Embed the question and pull the most similar chunks."""
    query_vector = embed_texts([question])[0]
    query_vector = np.expand_dims(query_vector, axis=0)
    results = store.search(query_vector, k=3)
    return [item["text"] for item, _ in results]


def interactive_session(store: FaissStore) -> None:
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

        context_chunks = retrieve_context(store, question)
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
    interactive_session(store)


if __name__ == "__main__":
    main()
