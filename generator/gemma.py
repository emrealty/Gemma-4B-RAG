"""Utility helpers for generating answers with Gemma via Ollama."""

from __future__ import annotations

import json
import os
from typing import Iterable

import requests

DEFAULT_GEN_MODEL = "gemma3:4b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def build_prompt(question: str, context_chunks: Iterable[str]) -> str:
    """Compose a concise instruction prompt with retrieved context."""
    context_block = "\n\n".join(context_chunks)
    prompt = (
        "Aşağıda kullanıcının sorusu ve ona yardımcı olabilecek bağlam parçaları var.\n"
        "Bağlam yeterli değilse dürüstçe belirt ve hayal etme.\n"
        f"---\nBağlam:\n{context_block}\n---\nSoru: {question}\nCevap:"
    )
    return prompt


def generate_with_gemma(prompt: str, model: str = DEFAULT_GEN_MODEL) -> str:
    """Call Ollama generate API and stream the model response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()
