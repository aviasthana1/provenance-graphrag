"""Environment-backed defaults for providers and graph storage."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROJECT_URL = "https://github.com/aviasthana1/provenance-graphrag"

MAX_CHUNK_CHARS = 2_000
CHUNKS_PER_LLM_CALL = 5
EMBEDDING_DIMENSION = 768
KNN_K = 6
KNN_MIN_SCORE = 0.85
ENTITY_MATCH_THRESHOLD = 0.80
ENTITY_EXTRACTION_MODEL = "mistralai/mistral-nemo"
COMMUNITY_SUMMARY_MODEL = "google/gemini-2.5-flash-lite"
COMMUNITY_MAX_LEVELS = 3
COMMUNITY_MIN_SIZE = 2
COMMUNITY_PROJECTION_NAME = "provenance_graphrag_communities"


def require_credentials(*names: str) -> None:
    """Raise one actionable error for missing runtime credentials."""
    missing = [name for name in names if not globals()[name]]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")
