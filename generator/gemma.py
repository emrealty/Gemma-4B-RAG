"""Utility helpers for generating answers with Gemma via Ollama."""

from __future__ import annotations

import json
import os
from typing import Iterable

import requests

DEFAULT_GEN_MODEL = "gemma3:4b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MISSING_ANSWER = "belgede böyle bir bilgi yok"


def build_prompt(question: str, context_chunks: Iterable[str]) -> str:
    """Compose a strict instruction prompt with retrieved context.

    Gereksinim: Bağlam boşsa veya soruyla açıkça ilişkili bilgi
    içermiyorsa, model "belgede böyle bir bilgi yok" demelidir.
    """
    context_block = "\n\n".join(context_chunks)
    prompt = (
        "Aşağıda kullanıcının sorusu ve ona yardımcı olabilecek bağlam parçaları var.\n"
        "Yalnızca bağlamdaki bilgilere dayanarak cevap ver.\n"
        "Bağlam soruyla ilgili anahtar kelimeler veya başlıklar içeriyorsa, oradaki bilgileri özetle.\n"
        "Bağlam boşsa ya da soruyu yanıtlayacak bilgiyi barındırmıyorsa, tahmin etme ve dış bilgi kullanma.\n"
        "Bu durumda cevabın tam olarak şu cümle olmalı: 'belgede böyle bir bilgi yok'.\n"
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
