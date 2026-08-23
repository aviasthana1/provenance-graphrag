"""
KNN similarity graph construction
"""

from typing import List, Tuple

from .config import KNN_K, KNN_MIN_SCORE
from .models import RelationshipEdge
from .utils import cosine_similarity, logger


def build_knn_similarity_graph(
    nodes_with_embeddings: List[Tuple[str, List[float], str]],  # [(id, embedding, type)]
    k: int = KNN_K,
    min_score: float = KNN_MIN_SCORE,
) -> List[RelationshipEdge]:
    """
    Build KNN similarity graph using embeddings.
    For each node, find k nearest neighbors with similarity >= min_score.

    Args:
        nodes_with_embeddings: List of (node_id, embedding, node_type) tuples
        k: Number of nearest neighbors to find
        min_score: Minimum similarity threshold

    Returns:
        List of SIMILAR relationships
    """
    logger.info(f"Building KNN similarity graph (k={k}, min_score={min_score})...")

    similarities = []

    for i, (node_id1, emb1, type1) in enumerate(nodes_with_embeddings):
        if i % 50 == 0 and i > 0:
            logger.info(f"   Progress: {i}/{len(nodes_with_embeddings)} nodes processed")

        neighbors = []

        for j, (node_id2, emb2, type2) in enumerate(nodes_with_embeddings):
            if i == j:
                continue

            score = cosine_similarity(emb1, emb2)

            if score >= min_score:
                neighbors.append((node_id2, score, type2))

        # Keep top k neighbors
        neighbors.sort(key=lambda x: x[1], reverse=True)
        for neighbor_id, score, neighbor_type in neighbors[:k]:
            similarities.append(
                RelationshipEdge(
                    from_id=node_id1,
                    to_id=neighbor_id,
                    type="SIMILAR",
                    confidence=score,
                    inferred=False,
                    properties={"score": score, "from_type": type1, "to_type": neighbor_type},
                )
            )

    logger.info(f"   Created {len(similarities)} KNN similarity relationships")
    return similarities
