"""
Entity resolution - matches new entities to existing ones
"""

from collections.abc import Mapping
from typing import Dict, List

from .config import ENTITY_MATCH_THRESHOLD
from .models import EntityNode
from .utils import cosine_similarity, fuzzy_match_score, logger

# Entity types that are semantically compatible (can be merged)
TYPE_COMPATIBILITY_GROUPS = {
    # Educational institutions
    frozenset({"EDUCATIONAL_INSTITUTION", "ORGANIZATION", "UNIVERSITY", "SCHOOL"}),
    # Companies/Organizations
    frozenset({"ORGANIZATION", "COMPANY", "CORPORATION", "EMPLOYER"}),
    # People
    frozenset({"PERSON", "INDIVIDUAL", "PROFESSIONAL"}),
    # Skills/Technologies
    frozenset({"SKILL", "TECHNOLOGY", "TOOL", "FRAMEWORK", "PROGRAMMING_LANGUAGE"}),
    # Locations
    frozenset({"LOCATION", "PLACE", "CITY", "COUNTRY", "REGION"}),
}


def _are_types_compatible(type1: str, type2: str) -> bool:
    """Check if two entity types are semantically compatible for merging."""
    if type1 == type2:
        return True

    type1_upper = type1.upper()
    type2_upper = type2.upper()

    for group in TYPE_COMPATIBILITY_GROUPS:
        if type1_upper in group and type2_upper in group:
            return True

    return False


def _get_canonical_name(name: str, aliases: Mapping[str, str]) -> str:
    """Get canonical form of a name, resolving aliases."""
    normalized = name.lower().strip()

    # Check for exact alias match
    if normalized in aliases:
        return aliases[normalized]

    # Check for partial matches (e.g., "Runna (Strava)" contains "runna")
    # Return the base name without parenthetical additions
    if "(" in normalized:
        base_name = normalized.split("(")[0].strip()
        if base_name in aliases:
            return aliases[base_name]
        return base_name

    return normalized


def resolve_entities_with_existing(
    new_entities: List[EntityNode],
    existing_entities: List[EntityNode],
    require_type_compatibility: bool = True,
    aliases: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """
    Resolve new entities against existing ones using fuzzy matching + embeddings.

    Features:
    - Type-aware matching (EDUCATIONAL_INSTITUTION can match ORGANIZATION)
    - Optional caller-supplied alias resolution
    - Combined fuzzy + embedding similarity scoring

    Args:
        new_entities: New entities to resolve
        existing_entities: Existing entities in the graph
        require_type_compatibility: If True, only match compatible types

    Returns mapping: new_entity_id -> existing_entity_id (for matches)
    """
    if not existing_entities:
        logger.info("No existing entities to resolve against")
        return {}

    resolution_map: Dict[str, str] = {}
    aliases = {key.lower().strip(): value.lower().strip() for key, value in (aliases or {}).items()}

    logger.info(
        f"Resolving {len(new_entities)} new entities against {len(existing_entities)} existing..."
    )

    for new_ent in new_entities:
        best_match_id = None
        best_score = 0.0
        best_match_name = ""
        best_match_type = ""

        # Get canonical name for the new entity
        new_canonical = _get_canonical_name(new_ent.name, aliases)

        for existing_ent in existing_entities:
            # Check type compatibility if required
            if require_type_compatibility and not _are_types_compatible(
                new_ent.type, existing_ent.type
            ):
                continue

            # Get canonical name for existing entity
            existing_canonical = _get_canonical_name(existing_ent.name, aliases)

            # 1. Text similarity (on canonical names)
            text_score = fuzzy_match_score(new_canonical, existing_canonical)

            # Boost score for exact canonical match
            if new_canonical == existing_canonical:
                text_score = 1.0

            # 2. Embedding similarity (if available)
            emb_score = 0.0
            if new_ent.embedding and existing_ent.embedding:
                emb_score = cosine_similarity(new_ent.embedding, existing_ent.embedding)

            # 3. Combined score (weighted: 40% text, 60% embedding)
            combined_score = 0.4 * text_score + 0.6 * emb_score

            # Type bonus: same type gets a small boost
            if new_ent.type == existing_ent.type:
                combined_score += 0.05

            if combined_score > best_score:
                best_score = combined_score
                best_match_id = existing_ent.id
                best_match_name = existing_ent.name
                best_match_type = existing_ent.type

        if best_score > ENTITY_MATCH_THRESHOLD:
            resolution_map[new_ent.id] = best_match_id
            type_info = (
                f" [{new_ent.type}->{best_match_type}]" if new_ent.type != best_match_type else ""
            )
            logger.info(
                f"   '{new_ent.name}' -> '{best_match_name}'{type_info} (score: {best_score:.2f})"
            )

    logger.info(f"Resolved {len(resolution_map)}/{len(new_entities)} entities to existing ones")
    return resolution_map
