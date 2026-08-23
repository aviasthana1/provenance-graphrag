"""
Entity extraction using Mistral Nemo via OpenRouter
"""

import json
import re
import time
import uuid
from typing import Dict, List, Tuple

import requests

from .config import (
    CHUNKS_PER_LLM_CALL,
    ENTITY_EXTRACTION_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    PROJECT_URL,
    require_credentials,
)
from .models import ChunkNode, EntityNode, RelationshipEdge
from .utils import fuzzy_match_score, logger, normalize_text

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting entities and relationships from text.

Analyze the following text chunks and extract:

1. ENTITIES: Important named entities (people, organizations, concepts, technologies, methods, locations, events)
   - Be specific and use the exact names from the text
   - Include brief descriptions

2. RELATIONSHIPS: How entities relate to each other
   - Use clear relationship types
   - Include confidence scores (0.0-1.0)

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanations):

{
  "entities": [
    {
      "name": "Neural Networks",
      "type": "CONCEPT",
      "description": "Computational models inspired by biological neural networks",
      "chunk_id": "chunk-id-here"
    }
  ],
  "relationships": [
    {
      "from": "Neural Networks",
      "to": "Deep Learning",
      "type": "PART_OF",
      "confidence": 0.95
    }
  ]
}

ENTITY TYPES:
- PERSON: Individual people
- ORGANIZATION: Companies, institutions, groups
- CONCEPT: Abstract ideas, theories, principles
- TECHNOLOGY: Tools, frameworks, software, hardware
- METHOD: Techniques, algorithms, approaches
- LOCATION: Places, regions
- EVENT: Specific occurrences, milestones

RELATIONSHIP TYPES:
- PART_OF: Component relationship
- USES: Usage relationship
- CREATED_BY: Authorship/creation
- ENABLES: Enabling/supporting
- RELATED_TO: General semantic relation
- IS_TYPE_OF: Subtype/instance
- PREREQUISITE_OF: Dependency relationship

CHUNKS TO ANALYZE:
"""


def extract_entities_batch(
    chunks: List[ChunkNode],
) -> Tuple[List[EntityNode], List[RelationshipEdge]]:
    """
    Extract entities from chunks using batched Mistral Nemo calls.
    Processes CHUNKS_PER_LLM_CALL chunks per API call for efficiency.
    """
    require_credentials("OPENROUTER_API_KEY")
    all_entities: Dict[str, EntityNode] = {}
    all_relationships: List[RelationshipEdge] = []

    logger.info(f"Extracting entities from {len(chunks)} chunks using {ENTITY_EXTRACTION_MODEL}...")
    logger.info(f"   Batch size: {CHUNKS_PER_LLM_CALL} chunks per LLM call")

    num_batches = (len(chunks) + CHUNKS_PER_LLM_CALL - 1) // CHUNKS_PER_LLM_CALL

    for batch_idx in range(0, len(chunks), CHUNKS_PER_LLM_CALL):
        batch = chunks[batch_idx : batch_idx + CHUNKS_PER_LLM_CALL]
        batch_num = (batch_idx // CHUNKS_PER_LLM_CALL) + 1

        logger.info(
            f"   Batch {batch_num}/{num_batches}: Processing chunks {batch_idx + 1}-{min(batch_idx + CHUNKS_PER_LLM_CALL, len(chunks))}"
        )

        # Build prompt with chunk context
        chunks_text = ""
        for j, chunk in enumerate(batch):
            chunks_text += f"\n\n--- CHUNK {j + 1} ---\n"
            chunks_text += f"Chunk ID: {chunk.id}\n"
            chunks_text += f"Topic: {chunk.heading_path}\n"
            chunks_text += f"Content:\n{chunk.content[:800]}\n"  # First 800 chars

        user_prompt = (
            ENTITY_EXTRACTION_PROMPT
            + chunks_text
            + "\n\nExtract entities and return ONLY the JSON:"
        )

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": PROJECT_URL,
                    "X-Title": "Provenance GraphRAG",
                },
                json={
                    "model": ENTITY_EXTRACTION_MODEL,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON (handle markdown code blocks)
            json_content = content.strip()
            if json_content.startswith("```"):
                json_content = re.sub(r"^```(?:json)?\n?", "", json_content)
                json_content = re.sub(r"\n?```$", "", json_content)

            result = json.loads(json_content)

            # Process entities
            entities_added = 0
            for ent_data in result.get("entities", []):
                entity_name = ent_data.get("name", "").strip()
                if not entity_name:
                    continue

                normalized_name = normalize_text(entity_name)

                # Check if entity already exists (fuzzy match for deduplication)
                existing = None
                for existing_id, existing_ent in all_entities.items():
                    score = fuzzy_match_score(normalized_name, existing_ent.normalized_name)
                    if score > 0.85:  # High threshold for within-document dedup
                        existing = existing_ent
                        break

                if existing:
                    # Merge into existing entity
                    chunk_id = ent_data.get("chunk_id", "")
                    if chunk_id and chunk_id not in existing.source_chunks:
                        existing.source_chunks.append(chunk_id)
                else:
                    # Create new entity
                    entity_id = str(uuid.uuid4())
                    entity = EntityNode(
                        id=entity_id,
                        name=entity_name,
                        type=ent_data.get("type", "CONCEPT"),
                        description=ent_data.get("description", ""),
                        source_chunks=[ent_data.get("chunk_id", "")],
                        normalized_name=normalized_name,
                    )
                    all_entities[entity_id] = entity
                    entities_added += 1

            # Process relationships
            entity_name_to_id = {ent.name.lower(): ent.id for ent in all_entities.values()}
            rels_added = 0

            for rel_data in result.get("relationships", []):
                from_name = rel_data.get("from", "").strip().lower()
                to_name = rel_data.get("to", "").strip().lower()

                from_id = entity_name_to_id.get(from_name)
                to_id = entity_name_to_id.get(to_name)

                if from_id and to_id and from_id != to_id:
                    relationship = RelationshipEdge(
                        from_id=from_id,
                        to_id=to_id,
                        type=rel_data.get("type", "RELATED_TO"),
                        confidence=rel_data.get("confidence", 0.9),
                        inferred=False,
                    )
                    all_relationships.append(relationship)
                    rels_added += 1

            logger.info(
                f"      -> Extracted {entities_added} new entities, {rels_added} relationships"
            )

        except json.JSONDecodeError as e:
            logger.error(f"   JSON parse error for batch {batch_num}: {e}")
            continue
        except Exception as e:
            logger.error(f"   Entity extraction failed for batch {batch_num}: {e}")
            continue

        time.sleep(0.5)  # Rate limiting

    logger.info(
        f"Extraction complete: {len(all_entities)} unique entities, {len(all_relationships)} relationships"
    )
    return list(all_entities.values()), all_relationships
