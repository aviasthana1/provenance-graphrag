"""
Inference algorithms for relationship discovery
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from .models import EntityNode, RelationshipEdge
from .utils import logger


def apply_transitive_inference(
    relationships: List[RelationshipEdge], entity_id_to_name: Dict[str, str]
) -> List[RelationshipEdge]:
    """
    Apply transitive inference: A→B→C implies A→C
    Only applies to non-inferred relationships to avoid infinite loops.
    """
    logger.info("Applying transitive inference...")

    # Build adjacency graph from direct relationships only
    graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for rel in relationships:
        if not rel.inferred:
            graph[rel.from_id].append((rel.to_id, rel.type))

    inferred = []
    existing_pairs = {(r.from_id, r.to_id) for r in relationships}

    # Find paths of length 2
    for start_node in graph:
        for mid_node, rel_type1 in graph[start_node]:
            if mid_node in graph:
                for end_node, rel_type2 in graph[mid_node]:
                    if (start_node, end_node) not in existing_pairs and start_node != end_node:
                        # Infer transitive relationship
                        mid_name = entity_id_to_name.get(mid_node, mid_node[:8])
                        inferred.append(
                            RelationshipEdge(
                                from_id=start_node,
                                to_id=end_node,
                                type=f"INDIRECTLY_{rel_type1}",
                                confidence=0.7,
                                inferred=True,
                                properties={"via": mid_node, "via_name": mid_name},
                            )
                        )
                        existing_pairs.add((start_node, end_node))

    logger.info(f"   Inferred {len(inferred)} transitive relationships")
    return inferred


def apply_lexical_similarity_inference(entities: List[EntityNode]) -> List[RelationshipEdge]:
    """
    Infer relationships based on lexical similarity (word overlap, containment)
    """
    logger.info("Applying lexical similarity inference...")

    inferred = []

    for i, ent1 in enumerate(entities):
        for ent2 in entities[i + 1 :]:
            words1 = set(ent1.normalized_name.split())
            words2 = set(ent2.normalized_name.split())

            if not words1 or not words2:
                continue

            # Word overlap
            overlap = len(words1 & words2) / max(len(words1), len(words2))

            if overlap > 0.5:  # Significant overlap
                inferred.append(
                    RelationshipEdge(
                        from_id=ent1.id,
                        to_id=ent2.id,
                        type="RELATES_TO",
                        confidence=overlap,
                        inferred=True,
                        properties={"method": "lexical_overlap", "overlap": overlap},
                    )
                )

            # Containment (one name contains the other)
            if len(ent1.normalized_name) > 3 and len(ent2.normalized_name) > 3:
                if ent1.normalized_name in ent2.normalized_name:
                    inferred.append(
                        RelationshipEdge(
                            from_id=ent1.id,
                            to_id=ent2.id,
                            type="IS_SUBTYPE_OF",
                            confidence=0.8,
                            inferred=True,
                            properties={"method": "containment"},
                        )
                    )
                elif ent2.normalized_name in ent1.normalized_name:
                    inferred.append(
                        RelationshipEdge(
                            from_id=ent2.id,
                            to_id=ent1.id,
                            type="IS_SUBTYPE_OF",
                            confidence=0.8,
                            inferred=True,
                            properties={"method": "containment"},
                        )
                    )

    logger.info(f"   Inferred {len(inferred)} lexical similarity relationships")
    return inferred
