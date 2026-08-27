"""
Community Detection and Report Generation for the Enhanced Knowledge Graph.

Uses Neo4j GDS Leiden algorithm for community detection, and LLM-generated
summaries for community reports. Communities are stored as Neo4j nodes with
IN_COMMUNITY and PARENT_COMMUNITY relationships.

Pipeline:
1. Create GDS graph projection (Entity nodes, weighted relationships)
2. Run gds.leiden.write() → writes communities property on entities
3. Create Community nodes + hierarchy
4. Generate LLM summaries (with caching via member hash)
5. Generate embeddings
6. Store everything in Neo4j with indexes
"""

import hashlib
import json
import re
import time
from typing import Dict, List, Optional

import requests

from .config import (
    COMMUNITY_MAX_LEVELS,
    COMMUNITY_MIN_SIZE,
    COMMUNITY_PROJECTION_NAME,
    COMMUNITY_SUMMARY_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    PROJECT_URL,
    require_credentials,
)
from .embeddings import embed_texts_batch
from .neo4j_client import Neo4jClientEnhanced
from .utils import logger

# =============================================================================
# LLM PROMPTS
# =============================================================================

LEAF_COMMUNITY_PROMPT = """You are an AI assistant analyzing a community of related entities in a knowledge graph.

Given the entities and relationships below that belong to the same community, generate a concise report.

Return ONLY valid JSON (no markdown, no extra text):
{{
    "title": "A short descriptive title (max 6 words)",
    "summary": "A natural language summary of this community - what it represents, how the entities are connected, and why it matters. 2-4 sentences.",
    "key_entities": ["entity1", "entity2", "entity3"],
    "key_relationships": ["entity1 WORKS_AT entity2", "entity3 USES entity4"],
    "importance_score": 7.5
}}

importance_score: Float 0-10. Higher = more central/important community.

Community members:
{community_info}
"""

PARENT_COMMUNITY_PROMPT = """You are an AI assistant synthesizing community summaries in a knowledge graph.

Given the following child community summaries that belong to the same parent community,
generate a combined report that synthesizes the information.

Return ONLY valid JSON (no markdown, no extra text):
{{
    "title": "A short descriptive title (max 6 words)",
    "summary": "A natural language summary that synthesizes all child communities. What is the overarching theme? 2-4 sentences.",
    "key_entities": ["entity1", "entity2", "entity3"],
    "key_relationships": ["entity1 WORKS_AT entity2"],
    "importance_score": 7.5
}}

importance_score: Float 0-10. Higher = more central/important community.

Child community summaries:
{community_info}
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _compute_member_hash(entity_ids: List[str]) -> str:
    """Deterministic hash of sorted entity IDs for caching."""
    sorted_ids = sorted(entity_ids)
    return hashlib.sha256(",".join(sorted_ids).encode()).hexdigest()[:16]


def _prepare_leaf_context(nodes: list, rels: list) -> str:
    """Format entity + relationship data for the LLM prompt."""
    lines = []
    lines.append("Entities:")
    for node in nodes:
        desc = f", description: {node['description']}" if node.get("description") else ""
        lines.append(f"- {node['name']} ({node.get('type', 'UNKNOWN')}){desc}")

    if rels:
        lines.append("\nRelationships:")
        for rel in rels:
            lines.append(f"- {rel['start']} --[{rel['type']}]--> {rel['end']}")

    return "\n".join(lines)


def _prepare_parent_context(child_summaries: List[Dict]) -> str:
    """Format child community summaries for the parent prompt."""
    lines = []
    for i, child in enumerate(child_summaries, 1):
        lines.append(
            f'Community {i} - "{child.get("title", "Untitled")}": {child.get("summary", "No summary")}'
        )
    return "\n".join(lines)


def _call_community_llm(prompt: str) -> Optional[Dict]:
    """Call OpenRouter chat/completions to generate community report."""
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": PROJECT_URL,
                "X-Title": "Enhanced KG Community Builder",
            },
            json={
                "model": COMMUNITY_SUMMARY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1500,
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

        return json.loads(json_content)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in community LLM response: {e}")
        return None
    except Exception as e:
        logger.error(f"Community LLM call failed: {e}")
        return None


# =============================================================================
# GDS GRAPH PROJECTION
# =============================================================================


def _drop_projection_if_exists(client: Neo4jClientEnhanced, projection_name: str):
    """Drop existing GDS graph projection."""
    with client.driver.session() as session:
        try:
            # Check if projection exists
            result = session.run("CALL gds.graph.exists($name) YIELD exists", name=projection_name)
            record = result.single()
            if record and record["exists"]:
                session.run("CALL gds.graph.drop($name)", name=projection_name)
                logger.info(f"   Dropped existing projection '{projection_name}'")
        except Exception as e:
            logger.warning(f"   Could not check/drop projection: {e}")


def _create_graph_projection(
    client: Neo4jClientEnhanced,
    corpus_id: str,
    projection_name: str,
) -> Dict:
    """Create an undirected weighted GDS graph projection from Entity nodes.
    Uses memory parameter required by Neo4j Aura GDS."""
    with client.driver.session() as session:
        result = session.run(
            """
            MATCH (source:Entity {corpus_id: $corpus_id})-[r]->(target:Entity {corpus_id: $corpus_id})
            WHERE NOT type(r) IN ['SIMILAR', 'MENTIONS', 'IN_COMMUNITY']
            WITH source, target, count(*) AS weight
            WITH gds.graph.project(
                $projection_name,
                source,
                target,
                {relationshipProperties: {weight: weight}},
                {undirectedRelationshipTypes: ['*'], memory: '2GB'}
            ) AS g
            RETURN g.graphName AS graph_name, g.nodeCount AS nodes, g.relationshipCount AS rels
        """,
            corpus_id=corpus_id,
            projection_name=projection_name,
        )

        record = result.single()
        if not record:
            return {"nodes": 0, "rels": 0}
        return {
            "graph_name": record["graph_name"],
            "nodes": record["nodes"],
            "rels": record["rels"],
        }


# =============================================================================
# LEIDEN COMMUNITY DETECTION
# =============================================================================


def _run_leiden(
    client: Neo4jClientEnhanced,
    projection_name: str,
    max_levels: int = COMMUNITY_MAX_LEVELS,
    min_community_size: int = COMMUNITY_MIN_SIZE,
) -> Dict:
    """Run GDS Leiden algorithm and write community assignments to Entity nodes."""
    with client.driver.session() as session:
        result = session.run(
            """
            CALL gds.leiden.write($projection_name, {
                writeProperty: 'communities',
                includeIntermediateCommunities: true,
                relationshipWeightProperty: 'weight',
                maxLevels: $max_levels,
                minCommunitySize: $min_community_size,
                randomSeed: 42
            })
            YIELD communityCount, modularity, modularities
            RETURN communityCount, modularity, modularities
        """,
            projection_name=projection_name,
            max_levels=max_levels,
            min_community_size=min_community_size,
        )

        record = result.single()
        if not record:
            return {"community_count": 0, "modularity": 0.0}
        return {
            "community_count": record["communityCount"],
            "modularity": record["modularity"],
            "modularities": record["modularities"],
        }


# =============================================================================
# COMMUNITY NODE CREATION
# =============================================================================


def _create_community_hierarchy(
    client: Neo4jClientEnhanced,
    corpus_id: str,
) -> int:
    """
    Create Community nodes from Leiden results on Entity.communities property.
    Creates IN_COMMUNITY and PARENT_COMMUNITY relationships.
    Returns count of communities created.
    """
    with client.driver.session() as session:
        # Create community nodes at each level and link entities + hierarchy
        # Following llm-graph-builder pattern: id = "<level>-<community_id>"
        session.run(
            """
            MATCH (e:Entity {corpus_id: $corpus_id})
            WHERE e.communities IS NOT NULL
            UNWIND range(0, size(e.communities) - 1, 1) AS index
            CALL {
                WITH e, index
                WITH e, index
                WHERE index = 0
                MERGE (c:Community {id: $corpus_id + '-' + toString(index) + '-' + toString(e.communities[index])})
                ON CREATE SET c.level = index, c.corpus_id = $corpus_id
                MERGE (e)-[:IN_COMMUNITY]->(c)
                RETURN count(*) AS count_0
            }
            CALL {
                WITH e, index
                WITH e, index
                WHERE index > 0
                MERGE (current:Community {id: $corpus_id + '-' + toString(index) + '-' + toString(e.communities[index])})
                ON CREATE SET current.level = index, current.corpus_id = $corpus_id
                MERGE (previous:Community {id: $corpus_id + '-' + toString(index - 1) + '-' + toString(e.communities[index - 1])})
                ON CREATE SET previous.level = index - 1, previous.corpus_id = $corpus_id
                MERGE (previous)-[:PARENT_COMMUNITY]->(current)
                RETURN count(*) AS count_1
            }
            RETURN count(*)
        """,
            corpus_id=corpus_id,
        )

        # Set community ranks (number of distinct sources connected)
        session.run(
            """
            MATCH (c:Community {corpus_id: $corpus_id})<-[:IN_COMMUNITY]-(e:Entity)<-[:MENTIONS]-(ch:Chunk)<-[:HAS_CHUNK]-(s:Source)
            WITH c, count(DISTINCT s) AS rank
            SET c.community_rank = rank
        """,
            corpus_id=corpus_id,
        )

        # Set community weights (entity count)
        session.run(
            """
            MATCH (c:Community {corpus_id: $corpus_id})<-[:IN_COMMUNITY]-(e:Entity)
            WITH c, count(DISTINCT e) AS entity_count
            SET c.weight = entity_count, c.entity_count = entity_count
        """,
            corpus_id=corpus_id,
        )

        # Set parent community weights via hierarchy
        session.run(
            """
            MATCH (c:Community {corpus_id: $corpus_id})
            WHERE c.level > 0
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY*]-(leaf:Community)<-[:IN_COMMUNITY]-(e:Entity)
            WITH c, count(DISTINCT e) AS entity_count
            SET c.weight = entity_count, c.entity_count = entity_count
        """,
            corpus_id=corpus_id,
        )

        # Count total communities
        result = session.run(
            """
            MATCH (c:Community {corpus_id: $corpus_id})
            RETURN count(c) AS total
        """,
            corpus_id=corpus_id,
        )
        return result.single()["total"]


# =============================================================================
# COMMUNITY REPORT GENERATION
# =============================================================================


def _fetch_leaf_community_info(
    client: Neo4jClientEnhanced,
    corpus_id: str,
) -> List[Dict]:
    """Fetch member entities and relationships for each leaf (level 0) community."""
    with client.driver.session() as session:
        # Fetch entities per community
        entities_result = session.run(
            """
            MATCH (c:Community {corpus_id: $corpus_id, level: 0})<-[:IN_COMMUNITY]-(e:Entity)
            WITH c.id AS communityId, collect({
                id: e.id, name: e.name, type: e.type, description: e.description
            }) AS nodes
            WHERE size(nodes) >= $min_size
            RETURN communityId, nodes
        """,
            corpus_id=corpus_id,
            min_size=COMMUNITY_MIN_SIZE,
        )

        community_data = {}
        for record in entities_result:
            community_data[record["communityId"]] = {
                "communityId": record["communityId"],
                "nodes": record["nodes"],
                "rels": [],
            }

        if not community_data:
            return []

        # Fetch internal relationships for each community
        # Get all entity IDs per community, then find relationships between them
        for comm_id, data in community_data.items():
            entity_ids = [n["id"] for n in data["nodes"]]
            rels_result = session.run(
                """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
                AND NOT type(r) IN ['SIMILAR', 'MENTIONS', 'IN_COMMUNITY']
                RETURN DISTINCT e1.name AS start, type(r) AS type, e2.name AS end
                LIMIT 20
            """,
                entity_ids=entity_ids,
            )
            data["rels"] = [dict(r) for r in rels_result]

        return list(community_data.values())


def _fetch_parent_community_info(
    client: Neo4jClientEnhanced,
    corpus_id: str,
) -> List[Dict]:
    """Fetch child community summaries for parent communities that need summaries."""
    with client.driver.session() as session:
        result = session.run(
            """
            MATCH (parent:Community {corpus_id: $corpus_id})<-[:PARENT_COMMUNITY]-(child:Community)
            WHERE parent.summary IS NULL AND child.summary IS NOT NULL
            WITH parent.id AS communityId, collect({
                title: child.title, summary: child.summary
            }) AS children
            RETURN communityId, children
        """,
            corpus_id=corpus_id,
        )
        return [dict(r) for r in result]


def _generate_leaf_summaries(
    client: Neo4jClientEnhanced,
    corpus_id: str,
    existing_hashes: Dict[str, Dict],
) -> int:
    """Generate LLM summaries for leaf communities. Returns count of communities summarized."""
    communities = _fetch_leaf_community_info(client, corpus_id)
    if not communities:
        logger.info("   No leaf communities to summarize")
        return 0

    summaries = []
    cached_count = 0

    for community in communities:
        entity_ids = [n["id"] for n in community["nodes"]]
        member_hash = _compute_member_hash(entity_ids)

        # Check cache
        if member_hash in existing_hashes:
            cached = existing_hashes[member_hash]
            summaries.append(
                {
                    "community": community["communityId"],
                    "title": cached["title"],
                    "summary": cached["summary"],
                    "importance_score": cached.get("importance_score", 5.0),
                    "key_entities": cached.get("key_entities", []),
                    "key_relationships": cached.get("key_relationships", []),
                    "member_hash": member_hash,
                }
            )
            cached_count += 1
            continue

        # Generate via LLM
        context = _prepare_leaf_context(community["nodes"], community["rels"])
        prompt = LEAF_COMMUNITY_PROMPT.format(community_info=context)
        result = _call_community_llm(prompt)

        if result:
            summaries.append(
                {
                    "community": community["communityId"],
                    "title": result.get("title", "Untitled Community"),
                    "summary": result.get("summary", ""),
                    "importance_score": result.get("importance_score", 5.0),
                    "key_entities": result.get("key_entities", []),
                    "key_relationships": result.get("key_relationships", []),
                    "member_hash": member_hash,
                }
            )
        else:
            # Fallback: generate simple title from entity names
            entity_names = [n["name"] for n in community["nodes"][:3]]
            summaries.append(
                {
                    "community": community["communityId"],
                    "title": " & ".join(entity_names),
                    "summary": f"A community of {len(community['nodes'])} related entities: {', '.join(n['name'] for n in community['nodes'])}.",
                    "importance_score": 5.0,
                    "key_entities": entity_names,
                    "key_relationships": [],
                    "member_hash": member_hash,
                }
            )

        time.sleep(0.5)  # Rate limiting

    if cached_count > 0:
        logger.info(f"   Reused {cached_count} cached community summaries")

    # Store summaries in Neo4j
    if summaries:
        with client.driver.session() as session:
            session.run(
                """
                UNWIND $data AS row
                MATCH (c:Community {id: row.community})
                SET c.title = row.title,
                    c.summary = row.summary,
                    c.importance_score = row.importance_score,
                    c.key_entities = row.key_entities,
                    c.key_relationships = row.key_relationships,
                    c.member_hash = row.member_hash
            """,
                data=summaries,
            )

    return len(summaries)


def _generate_parent_summaries(
    client: Neo4jClientEnhanced,
    corpus_id: str,
) -> int:
    """Generate LLM summaries for parent communities. Returns count summarized."""
    parents = _fetch_parent_community_info(client, corpus_id)
    if not parents:
        logger.info("   No parent communities to summarize")
        return 0

    summaries = []
    for parent in parents:
        context = _prepare_parent_context(parent["children"])
        prompt = PARENT_COMMUNITY_PROMPT.format(community_info=context)
        result = _call_community_llm(prompt)

        if result:
            summaries.append(
                {
                    "community": parent["communityId"],
                    "title": result.get("title", "Untitled Group"),
                    "summary": result.get("summary", ""),
                    "importance_score": result.get("importance_score", 5.0),
                    "key_entities": result.get("key_entities", []),
                    "key_relationships": result.get("key_relationships", []),
                }
            )
        else:
            # Fallback
            child_titles = [c.get("title", "?") for c in parent["children"]]
            summaries.append(
                {
                    "community": parent["communityId"],
                    "title": "Group: " + ", ".join(child_titles[:2]),
                    "summary": f"A parent community encompassing {len(parent['children'])} sub-communities: {', '.join(child_titles)}.",
                    "importance_score": 5.0,
                    "key_entities": [],
                    "key_relationships": [],
                }
            )

        time.sleep(0.5)

    if summaries:
        with client.driver.session() as session:
            session.run(
                """
                UNWIND $data AS row
                MATCH (c:Community {id: row.community})
                SET c.title = row.title,
                    c.summary = row.summary,
                    c.importance_score = row.importance_score,
                    c.key_entities = row.key_entities,
                    c.key_relationships = row.key_relationships
            """,
                data=summaries,
            )

    return len(summaries)


# =============================================================================
# COMMUNITY EMBEDDINGS
# =============================================================================


def _generate_and_store_embeddings(
    client: Neo4jClientEnhanced,
    corpus_id: str,
) -> int:
    """Generate embeddings for all communities with summaries and store in Neo4j."""
    with client.driver.session() as session:
        result = session.run(
            """
            MATCH (c:Community {corpus_id: $corpus_id})
            WHERE c.summary IS NOT NULL AND c.embedding IS NULL
            RETURN c.id AS id, c.title AS title, c.summary AS summary
        """,
            corpus_id=corpus_id,
        )
        communities = [dict(r) for r in result]

    if not communities:
        return 0

    texts = [f"{c['title']}\n\n{c['summary']}" for c in communities]
    embeddings = embed_texts_batch(texts)

    with client.driver.session() as session:
        for community, embedding in zip(communities, embeddings):
            if embedding and any(v != 0 for v in embedding):
                session.run(
                    """
                    MATCH (c:Community {id: $id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                    RETURN c
                """,
                    id=community["id"],
                    embedding=embedding,
                )

    return len(communities)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def build_communities(corpus_id: str) -> Dict:
    """
    Main orchestrator: detect communities, generate reports, store in Neo4j.

    Pipeline:
    1. Check entity count (skip if < 3)
    2. Create GDS graph projection
    3. Run Leiden algorithm
    4. Create Community nodes + hierarchy
    5. Generate LLM summaries (with caching)
    6. Generate embeddings
    7. Cleanup

    Returns dict with stats about communities created.
    """
    require_credentials("NEO4J_URI", "NEO4J_PASSWORD", "OPENROUTER_API_KEY", "GEMINI_API_KEY")
    client = Neo4jClientEnhanced(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        if not client.test_connection():
            raise RuntimeError("Failed to connect to Neo4j")

        # Check entity count
        with client.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {corpus_id: $corpus_id})
                RETURN count(e) AS entity_count
            """,
                corpus_id=corpus_id,
            )
            entity_count = result.single()["entity_count"]

        if entity_count < 3:
            logger.info(f"   Only {entity_count} entities — skipping community detection")
            return {"communities_created": 0, "reason": "too_few_entities"}

        logger.info(f"   Found {entity_count} entities for community detection")

        # Fetch existing community hashes for caching before clearing
        existing_hashes = client.fetch_existing_community_hashes(corpus_id)
        if existing_hashes:
            logger.info(f"   Cached {len(existing_hashes)} existing community hashes")

        # Clear old communities
        deleted = client.clear_communities(corpus_id)
        if deleted > 0:
            logger.info(f"   Cleared {deleted} old Community nodes")

        # Ensure indexes
        client.ensure_indexes()

        # Create GDS graph projection
        projection_name = f"{COMMUNITY_PROJECTION_NAME}_{corpus_id}"
        _drop_projection_if_exists(client, projection_name)

        logger.info("   Creating GDS graph projection...")
        projection = _create_graph_projection(client, corpus_id, projection_name)
        logger.info(
            f"   Projection: {projection.get('nodes', 0)} nodes, {projection.get('rels', 0)} relationships"
        )

        if projection.get("nodes", 0) < 3:
            logger.info("   Too few nodes in projection — skipping Leiden")
            _drop_projection_if_exists(client, projection_name)
            return {"communities_created": 0, "reason": "too_few_projected_nodes"}

        # Run Leiden
        logger.info("   Running Leiden community detection...")
        leiden_result = _run_leiden(client, projection_name)
        logger.info(
            f"   Leiden: {leiden_result.get('community_count', 0)} communities, "
            f"modularity={leiden_result.get('modularity', 0):.4f}"
        )

        # Create Community nodes + hierarchy
        logger.info("   Creating Community nodes and hierarchy...")
        total_communities = _create_community_hierarchy(client, corpus_id)
        logger.info(f"   Created {total_communities} Community nodes")

        # Generate leaf community summaries
        logger.info("   Generating leaf community summaries via LLM...")
        leaf_count = _generate_leaf_summaries(client, corpus_id, existing_hashes)
        logger.info(f"   Summarized {leaf_count} leaf communities")

        # Generate parent community summaries
        logger.info("   Generating parent community summaries via LLM...")
        parent_count = _generate_parent_summaries(client, corpus_id)
        logger.info(f"   Summarized {parent_count} parent communities")

        # Generate embeddings
        logger.info("   Generating community embeddings...")
        embedded_count = _generate_and_store_embeddings(client, corpus_id)
        logger.info(f"   Embedded {embedded_count} communities")

        # Cleanup: drop GDS projection
        _drop_projection_if_exists(client, projection_name)

        return {
            "communities_created": total_communities,
            "leaf_summaries": leaf_count,
            "parent_summaries": parent_count,
            "embeddings_generated": embedded_count,
            "leiden_community_count": leiden_result.get("community_count", 0),
            "leiden_modularity": leiden_result.get("modularity", 0.0),
        }

    except Exception as e:
        logger.error(f"Community building failed: {e}")
        # Try to clean up projection on error
        try:
            projection_name = f"{COMMUNITY_PROJECTION_NAME}_{corpus_id}"
            _drop_projection_if_exists(client, projection_name)
        except Exception:
            pass
        raise

    finally:
        client.close()
