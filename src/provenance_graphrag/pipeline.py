"""
Main pipeline for building enhanced knowledge graphs
"""

import uuid
from typing import Dict, List, Tuple

from .chunker import split_content_into_chunks
from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, require_credentials
from .embeddings import embed_texts_batch
from .entity_extraction import extract_entities_batch
from .entity_resolution import resolve_entities_with_existing
from .inference import apply_lexical_similarity_inference, apply_transitive_inference
from .models import ChunkNode, TopicNode
from .neo4j_client import Neo4jClientEnhanced
from .parser import flatten_heading_tree, parse_heading_hierarchy
from .similarity import build_knn_similarity_graph
from .utils import logger, print_heading_tree


def build_enhanced_knowledge_graph(
    text: str, corpus_id: str, source_title: str = "Document"
) -> Dict:
    """
    Build enhanced knowledge graph with entity extraction and inference.

    Args:
        text: Markdown/text content to process
        corpus_id: Unique identifier for the corpus
        source_title: Title for the source document

    Returns:
        Dictionary with build results and statistics
    """
    require_credentials("NEO4J_URI", "NEO4J_PASSWORD", "OPENROUTER_API_KEY", "GEMINI_API_KEY")
    source_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{corpus_id}:{source_title}"))

    logger.info("=" * 70)
    logger.info("ENHANCED KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 70)
    logger.info(f"Corpus ID: {corpus_id}")
    logger.info(f"Source: {source_title}")
    logger.info(f"Text length: {len(text)} characters")
    logger.info("")

    # =========================================================================
    # STEP 1: Parse document hierarchy
    # =========================================================================
    logger.info("STEP 1: Parsing document hierarchy...")
    root_headings = parse_heading_hierarchy(text)
    all_headings = flatten_heading_tree(root_headings)
    logger.info(f"Found {len(all_headings)} headings:")
    print_heading_tree(root_headings)
    logger.info("")

    if not all_headings:
        logger.warning("No headings found in document")
        return {"success": False, "error": "No headings found"}

    # =========================================================================
    # STEP 2: Create Topics and Chunks
    # =========================================================================
    logger.info("STEP 2: Creating Topics and Chunks...")

    topics: List[TopicNode] = []
    all_chunks: List[ChunkNode] = []
    chunk_index = 0

    for heading in all_headings:
        # Create topic
        topic = TopicNode(
            id=heading.id,
            corpus_id=corpus_id,
            title=heading.title,
            level=heading.level,
            summary="",
            content=heading.content,
            parent_id=heading.parent_id,
            position=heading.position,
        )
        topics.append(topic)

        # Create chunks
        if heading.content.strip():
            # Build heading path
            path_parts = [heading.title]
            current = heading
            while current.parent_id:
                parent = next((h for h in all_headings if h.id == current.parent_id), None)
                if parent:
                    path_parts.insert(0, parent.title)
                    current = parent
                else:
                    break

            heading_path = " > ".join(path_parts)

            chunks = split_content_into_chunks(
                heading.content, heading.id, source_id, corpus_id, heading_path, chunk_index
            )
            all_chunks.extend(chunks)
            chunk_index += len(chunks)

    logger.info(f"Created {len(topics)} Topics and {len(all_chunks)} Chunks")
    logger.info("")

    # =========================================================================
    # STEP 3: Extract Entities (BATCH with Mistral Nemo)
    # =========================================================================
    logger.info("STEP 3: Extracting entities from chunks...")
    new_entities, entity_relationships = extract_entities_batch(all_chunks)
    logger.info("")

    # =========================================================================
    # STEP 3.5: Generate embeddings for new entities (needed for resolution)
    # =========================================================================
    logger.info("STEP 3.5: Generating embeddings for new entities...")
    if new_entities:
        entity_texts = [f"{e.name}\n{e.description}" for e in new_entities]
        entity_embeddings = embed_texts_batch(entity_texts)
        for entity, embedding in zip(new_entities, entity_embeddings):
            entity.embedding = embedding
    logger.info("")

    # =========================================================================
    # STEP 3.6: Check for existing entities and resolve
    # =========================================================================
    logger.info("STEP 3.6: Resolving entities with existing graph...")

    # Initialize variables
    existing_entities = []
    resolution_map = {}
    entities_to_create = []

    # Connect to Neo4j early to check for existing entities
    client = Neo4jClientEnhanced(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        if not client.test_connection():
            raise RuntimeError("Failed to connect to Neo4j")

        # Fetch existing entities
        existing_entities = client.fetch_all_entities(corpus_id)
        logger.info(f"   Found {len(existing_entities)} existing entities in corpus")

        # Resolve new entities against existing ones
        if existing_entities:
            resolution_map = resolve_entities_with_existing(new_entities, existing_entities)

            # Apply resolution: merge resolved entities
            for new_ent in new_entities:
                if new_ent.id in resolution_map:
                    # This entity matches an existing one - update the existing
                    existing_id = resolution_map[new_ent.id]
                    existing_ent = next((e for e in existing_entities if e.id == existing_id), None)
                    if existing_ent:
                        # Merge source chunks
                        for chunk_id in new_ent.source_chunks:
                            if chunk_id not in existing_ent.source_chunks:
                                existing_ent.source_chunks.append(chunk_id)
                        # Update entity relationships to use existing entity ID
                        for rel in entity_relationships:
                            if rel.from_id == new_ent.id:
                                rel.from_id = existing_id
                            if rel.to_id == new_ent.id:
                                rel.to_id = existing_id
                else:
                    # New entity - will be created
                    entities_to_create.append(new_ent)

            # Combine: existing entities (potentially updated) + new entities
            entities = existing_entities + entities_to_create
            logger.info(
                f"   Resolved: {len(resolution_map)} matches, {len(entities_to_create)} new entities"
            )
        else:
            entities = new_entities
            entities_to_create = new_entities
            logger.info("   No existing entities - all entities are new")

        logger.info("")

    except Exception as e:
        logger.error(f"Entity resolution failed: {e}")
        entities = new_entities
        entities_to_create = new_entities
        logger.info("Proceeding with all entities as new")
        logger.info("")

    # =========================================================================
    # STEP 4: Generate Embeddings (Topics and Chunks)
    # =========================================================================
    logger.info("STEP 4: Generating embeddings for Topics and Chunks...")

    # Topics
    topic_texts = [f"{t.title}\n\n{t.content[:500]}" for t in topics]
    topic_embeddings = embed_texts_batch(topic_texts)
    for topic, embedding in zip(topics, topic_embeddings):
        topic.embedding = embedding

    # Chunks
    if all_chunks:
        chunk_texts = [c.content for c in all_chunks]
        chunk_embeddings = embed_texts_batch(chunk_texts)
        for chunk, embedding in zip(all_chunks, chunk_embeddings):
            chunk.embedding = embedding

    logger.info("")

    # =========================================================================
    # STEP 5: Build KNN Similarity Graph
    # =========================================================================
    logger.info("STEP 5: Building KNN similarity graphs...")

    # Prepare nodes with embeddings
    nodes_with_emb: List[Tuple[str, List[float], str]] = []

    # Add topics
    for topic in topics:
        if topic.embedding:
            nodes_with_emb.append((topic.id, topic.embedding, "Topic"))

    # Add entities
    for entity in entities:
        if entity.embedding:
            nodes_with_emb.append((entity.id, entity.embedding, "Entity"))

    # Add chunks
    for chunk in all_chunks:
        if chunk.embedding:
            nodes_with_emb.append((chunk.id, chunk.embedding, "Chunk"))

    knn_relationships = build_knn_similarity_graph(nodes_with_emb)
    logger.info("")

    # =========================================================================
    # STEP 6: Apply Inference Algorithms
    # =========================================================================
    logger.info("STEP 6: Applying inference algorithms...")

    # Build entity ID to name mapping
    entity_id_to_name = {e.id: e.name for e in entities}

    # Transitive inference
    transitive_rels = apply_transitive_inference(entity_relationships, entity_id_to_name)

    # Lexical similarity
    lexical_rels = apply_lexical_similarity_inference(entities)

    # Combine all relationships
    all_relationships = entity_relationships + knn_relationships + transitive_rels + lexical_rels

    logger.info(f"Total relationships: {len(all_relationships)}")
    logger.info("")

    # =========================================================================
    # STEP 7: Write to Neo4j
    # =========================================================================
    logger.info("STEP 7: Writing to Neo4j...")

    try:
        logger.info("Connected to Neo4j")

        client.ensure_indexes()

        # Create Source
        client.create_source(source_id, corpus_id, source_title)
        logger.info("Created Source node")

        # Create Topics
        for topic in topics:
            client.create_topic(topic)
        logger.info(f"Created {len(topics)} Topic nodes")

        # Create Chunks
        for chunk in all_chunks:
            client.create_chunk(chunk)
        logger.info(f"Created {len(all_chunks)} Chunk nodes")

        # Create NEW Entities only
        entities_updated = 0
        existing_entity_ids = {e.id for e in existing_entities}

        for entity in entities:
            if entity.id in existing_entity_ids:
                entities_updated += 1
            else:
                client.create_entity(entity, corpus_id)

        logger.info(
            f"Created {len(entities_to_create)} new Entity nodes, updated {entities_updated} existing"
        )

        # Create Topic relationships
        for topic in topics:
            client.create_source_has_topic(source_id, topic.id)
            if topic.parent_id:
                client.create_has_subtopic(topic.parent_id, topic.id)

        logger.info("Created Topic relationships")

        # Create Chunk relationships
        for chunk in all_chunks:
            client.create_topic_has_chunk(chunk.topic_id, chunk.id, chunk.chunk_index)
            client.create_source_has_chunk(source_id, chunk.id, chunk.chunk_index)

        # Create NEXT_CHUNK
        for i in range(len(all_chunks) - 1):
            client.create_next_chunk(all_chunks[i].id, all_chunks[i + 1].id)

        logger.info("Created Chunk relationships")

        # Create Chunk-MENTIONS-Entity relationships
        for entity in entities:
            for chunk_id in entity.source_chunks:
                if chunk_id:
                    try:
                        client.create_chunk_mentions_entity(chunk_id, entity.id)
                    except Exception as exc:
                        logger.debug("Could not create provenance edge: %s", exc)

        logger.info("Created MENTIONS relationships")

        # Create all other relationships
        logger.info(f"Creating {len(all_relationships)} entity/similarity relationships...")
        for i, rel in enumerate(all_relationships):
            if i % 100 == 0 and i > 0:
                logger.info(f"   Progress: {i}/{len(all_relationships)}")
            try:
                client.create_relationship(rel)
            except Exception as e:
                logger.warning(f"Failed to create relationship: {e}")

        logger.info("Created all relationships")

        # Get final stats
        stats = client.get_stats(corpus_id)

    finally:
        client.close()

    logger.info("")
    logger.info("=" * 70)
    logger.info("BUILD COMPLETE!")
    logger.info("=" * 70)
    logger.info("This Document:")
    logger.info(f"   Topics: {len(topics)}")
    logger.info(f"   Chunks: {len(all_chunks)}")
    logger.info(f"   New Entities: {len(entities_to_create)}")
    logger.info(f"   Resolved Entities: {len(resolution_map)}")
    logger.info(f"   Relationships: {len(all_relationships)}")
    logger.info("")
    logger.info("Total Corpus Stats:")
    logger.info(f"   Sources: {stats.get('sources', '?')}")
    logger.info(f"   Topics: {stats.get('topics', '?')}")
    logger.info(f"   Chunks: {stats.get('chunks', '?')}")
    logger.info(f"   Entities: {stats.get('entities', '?')}")
    logger.info("")

    return {
        "success": True,
        "source_id": source_id,
        "corpus_id": corpus_id,
        "topics_created": len(topics),
        "chunks_created": len(all_chunks),
        "entities_created": len(entities_to_create),
        "entities_resolved": len(resolution_map),
        "relationships_created": len(all_relationships),
        "stats": stats,
    }
