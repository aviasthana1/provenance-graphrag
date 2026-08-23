"""
Embedding generation using Gemini API
"""

import time
from typing import List

import requests

from .config import (
    EMBEDDING_DIMENSION,
    GEMINI_API_KEY,
    GEMINI_API_URL,
    require_credentials,
)
from .utils import logger, normalize_embedding


def embed_single_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Generate embedding for single text using Gemini"""
    require_credentials("GEMINI_API_KEY")
    if not text.strip():
        return [0.0] * EMBEDDING_DIMENSION

    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    # Truncate very long texts
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars]

    body = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": EMBEDDING_DIMENSION,
    }

    for attempt in range(3):
        try:
            response = requests.post(url, json=body, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ValueError(
                    f"Gemini API error: {data['error'].get('message', 'Unknown error')}"
                )

            if "embedding" not in data or "values" not in data["embedding"]:
                raise ValueError("No embedding values in response")

            return normalize_embedding(data["embedding"]["values"])

        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1 * (attempt + 1))

    logger.error("Failed to generate embedding, returning zeros")
    return [0.0] * EMBEDDING_DIMENSION


def embed_texts_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    require_credentials("GEMINI_API_KEY")
    logger.info(f"Generating embeddings for {len(texts)} texts...")

    embeddings = []
    for i, text in enumerate(texts):
        if i > 0 and i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(texts)} embeddings")
        embedding = embed_single_text(text, task_type)
        embeddings.append(embedding)
        time.sleep(0.1)  # Rate limiting

    return embeddings
