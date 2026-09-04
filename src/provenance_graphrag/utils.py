"""
Utility functions for the Provenance GraphRAG
"""

import logging
import math
from difflib import SequenceMatcher
from typing import List

from .models import HeadingNode

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEXT UTILITIES
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for entity resolution"""
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
    }
    words = text.lower().split()
    normalized = " ".join(w for w in words if w not in stopwords and len(w) > 1)
    return normalized if normalized else text.lower()


def fuzzy_match_score(str1: str, str2: str) -> float:
    """Calculate fuzzy match score between two strings (0-1)"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================


def normalize_embedding(embedding: List[float]) -> List[float]:
    """L2 normalize embedding to unit length"""
    norm = math.sqrt(sum(v * v for v in embedding))
    if norm == 0:
        return embedding
    return [v / norm for v in embedding]


def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Calculate cosine similarity between two normalized embeddings"""
    if not emb1 or not emb2 or len(emb1) != len(emb2):
        return 0.0
    return sum(a * b for a, b in zip(emb1, emb2))


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================


def print_heading_tree(roots: List[HeadingNode], indent: int = 0):
    """Pretty print heading tree"""
    for node in roots:
        prefix = "  " * indent
        logger.info(f"{prefix}{'#' * node.level} {node.title} [{len(node.content)} chars]")
        print_heading_tree(node.children, indent + 1)
