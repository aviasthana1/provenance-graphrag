"""
Neo4j client for knowledge graph operations
"""

import json
import re
from typing import Dict, List

from neo4j import GraphDatabase

from .config import EMBEDDING_DIMENSION
from .models import ChunkNode, CommunityNode, EntityNode, RelationshipEdge, TopicNode
from .utils import logger

_RELATIONSHIP_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def validate_relationship_type(value: str) -> str:
    """Validate a relationship type before interpolation into Cypher."""
    normalized = value.strip().upper().replace(" ", "_").replace("-", "_")
    if not _RELATIONSHIP_TYPE.fullmatch(normalized):
        raise ValueError(f"Invalid relationship type: {value!r}")
    return normalized


class Neo4jClientEnhanced:
    """Enhanced Neo4j client with entity support and full-text indexes"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connecting to Neo4j...")

    def close(self):
        if self.driver:
            self.driver.close()

    def test_connection(self) -> bool:
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def ensure_indexes(self):
        """Create all necessary indexes (vector, regular, full-text)"""
        logger.info("Ensuring indexes...")

        with self.driver.session() as session:
            # Vector indexes
            vector_indexes = [
                ("chunk_embedding", "Chunk", "embedding"),
                ("topic_embedding", "Topic", "embedding"),
                ("entity_embedding", "Entity", "embedding"),
                ("community_embedding", "Community", "embedding"),
            ]

            for name, label, prop in vector_indexes:
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX {name} IF NOT EXISTS
                        FOR (n:{label}) ON n.{prop}
                        OPTIONS {{ indexConfig: {{
                            `vector.dimensions`: {EMBEDDING_DIMENSION},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info(f"   Vector index: {name}")
                except Exception as e:
                    logger.warning(f"   Vector index {name} exists or error: {e}")

            # Regular indexes
            regular_indexes = [
                ("source_corpus", "Source", "corpus_id"),
                ("chunk_corpus", "Chunk", "corpus_id"),
                ("topic_corpus", "Topic", "corpus_id"),
                ("entity_corpus", "Entity", "corpus_id"),
                ("entity_name", "Entity", "name"),
                ("entity_type", "Entity", "type"),
                ("community_corpus", "Community", "corpus_id"),
            ]

            for name, label, prop in regular_indexes:
                try:
                    session.run(f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
                    logger.info(f"   Regular index: {name}")
                except Exception:
                    pass

            # Full-text indexes
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                    FOR (e:Entity) ON EACH [e.name, e.description]
                """)
                logger.info("   Full-text index: entity_fulltext")
            except Exception:
                pass

            try:
                session.run("""
                    CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
                    FOR (c:Chunk) ON EACH [c.content]
                """)
                logger.info("   Full-text index: chunk_fulltext")
            except Exception:
                pass

            try:
                session.run("""
                    CREATE FULLTEXT INDEX community_fulltext IF NOT EXISTS
                    FOR (c:Community) ON EACH [c.title, c.summary]
                """)
                logger.info("   Full-text index: community_fulltext")
            except Exception:
                pass

        logger.info("Indexes ready")

    def clear_corpus(self, corpus_id: str) -> int:
        """Delete all nodes for a corpus"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n) WHERE n.corpus_id = $corpus_id
                DETACH DELETE n
                RETURN count(n) AS deleted
            """,
                corpus_id=corpus_id,
            )
            return result.single()["deleted"]

    def create_source(self, source_id: str, corpus_id: str, title: str):
        """Create Source node"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:Source {id: $id})
                SET s.corpus_id = $corpus_id,
                    s.title = $title,
                    s.type = 'markdown',
                    s.created_at = datetime()
            """,
                id=source_id,
                corpus_id=corpus_id,
                title=title,
            )

    def create_topic(self, topic: TopicNode):
        """Create Topic node with embedding"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (t:Topic {id: $id})
                SET t.corpus_id = $corpus_id,
                    t.title = $title,
                    t.level = $level,
                    t.summary = $summary,
                    t.content = $content,
                    t.parent_id = $parent_id,
                    t.position = $position,
                    t.created_at = datetime()
            """,
                id=topic.id,
                corpus_id=topic.corpus_id,
                title=topic.title,
                level=topic.level,
                summary=topic.summary,
                content=topic.content[:5000],
                parent_id=topic.parent_id,
                position=topic.position,
            )

            if topic.embedding and any(v != 0 for v in topic.embedding):
                session.run(
                    """
                    MATCH (t:Topic {id: $id})
                    CALL db.create.setNodeVectorProperty(t, 'embedding', $embedding)
                    RETURN t
                """,
                    id=topic.id,
                    embedding=topic.embedding,
                )

    def create_chunk(self, chunk: ChunkNode):
        """Create Chunk node with embedding"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.source_id = $source_id,
                    c.corpus_id = $corpus_id,
                    c.topic_id = $topic_id,
                    c.content = $content,
                    c.chunk_index = $chunk_index,
                    c.heading_path = $heading_path,
                    c.created_at = datetime()
            """,
                id=chunk.id,
                source_id=chunk.source_id,
                corpus_id=chunk.corpus_id,
                topic_id=chunk.topic_id,
                content=chunk.content[:5000],
                chunk_index=chunk.chunk_index,
                heading_path=chunk.heading_path,
            )

            if chunk.embedding and any(v != 0 for v in chunk.embedding):
                session.run(
                    """
                    MATCH (c:Chunk {id: $id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                    RETURN c
                """,
                    id=chunk.id,
                    embedding=chunk.embedding,
                )

    def create_entity(self, entity: EntityNode, corpus_id: str):
        """Create Entity node with embedding"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.corpus_id = $corpus_id,
                    e.name = $name,
                    e.type = $type,
                    e.description = $description,
                    e.normalized_name = $normalized_name,
                    e.created_at = datetime()
            """,
                id=entity.id,
                corpus_id=corpus_id,
                name=entity.name,
                type=entity.type,
                description=entity.description[:1000],
                normalized_name=entity.normalized_name,
            )

            if entity.embedding and any(v != 0 for v in entity.embedding):
                session.run(
                    """
                    MATCH (e:Entity {id: $id})
                    CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                    RETURN e
                """,
                    id=entity.id,
                    embedding=entity.embedding,
                )

    def create_chunk_mentions_entity(self, chunk_id: str, entity_id: str):
        """Create MENTIONS relationship from Chunk to Entity"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
            """,
                chunk_id=chunk_id,
                entity_id=entity_id,
            )

    def create_relationship(self, rel: RelationshipEdge):
        """Create relationship with properties"""
        with self.driver.session() as session:
            relationship_type = validate_relationship_type(rel.type)
            query = f"""
                MATCH (from {{id: $from_id}})
                MATCH (to {{id: $to_id}})
                MERGE (from)-[r:`{relationship_type}`]->(to)
                SET r.confidence = $confidence,
                    r.inferred = $inferred,
                    r.properties = $properties
                RETURN r
            """

            try:
                session.run(
                    query,
                    from_id=rel.from_id,
                    to_id=rel.to_id,
                    confidence=rel.confidence,
                    inferred=rel.inferred,
                    properties=json.dumps(rel.properties),
                )
            except Exception as e:
                logger.warning(f"Failed to create relationship {rel.type}: {e}")

    def create_source_has_topic(self, source_id: str, topic_id: str):
        """Create HAS_TOPIC relationship"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:Source {id: $source_id})
                MATCH (t:Topic {id: $topic_id})
                MERGE (s)-[:HAS_TOPIC]->(t)
            """,
                source_id=source_id,
                topic_id=topic_id,
            )

    def create_source_about_entity(self, source_id: str, entity_id: str):
        """Create SOURCE_ABOUT relationship from Source to an Entity (e.g., Person)"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:Source {id: $source_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (s)-[:SOURCE_ABOUT]->(e)
            """,
                source_id=source_id,
                entity_id=entity_id,
            )

    def create_has_subtopic(self, parent_id: str, child_id: str):
        """Create HAS_SUBTOPIC relationship"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (parent:Topic {id: $parent_id})
                MATCH (child:Topic {id: $child_id})
                MERGE (parent)-[:HAS_SUBTOPIC]->(child)
            """,
                parent_id=parent_id,
                child_id=child_id,
            )

    def create_topic_has_chunk(self, topic_id: str, chunk_id: str, position: int):
        """Create HAS_CHUNK relationship from Topic to Chunk"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (t:Topic {id: $topic_id})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (t)-[r:HAS_CHUNK]->(c)
                SET r.position = $position
            """,
                topic_id=topic_id,
                chunk_id=chunk_id,
                position=position,
            )

    def create_source_has_chunk(self, source_id: str, chunk_id: str, position: int):
        """Create HAS_CHUNK relationship from Source to Chunk"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:Source {id: $source_id})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (s)-[r:HAS_CHUNK]->(c)
                SET r.position = $position
            """,
                source_id=source_id,
                chunk_id=chunk_id,
                position=position,
            )

    def create_next_chunk(self, from_id: str, to_id: str):
        """Create NEXT_CHUNK relationship"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c1:Chunk {id: $from_id})
                MATCH (c2:Chunk {id: $to_id})
                MERGE (c1)-[:NEXT_CHUNK]->(c2)
            """,
                from_id=from_id,
                to_id=to_id,
            )

    def fetch_all_entities(self, corpus_id: str) -> List[EntityNode]:
        """Fetch all entities for a corpus"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {corpus_id: $corpus_id})
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       e.description AS description, e.normalized_name AS normalized_name
            """,
                corpus_id=corpus_id,
            )

            entities = []
            for record in result:
                entity = EntityNode(
                    id=record["id"],
                    name=record["name"],
                    type=record["type"] or "CONCEPT",
                    description=record["description"] or "",
                    normalized_name=record["normalized_name"] or "",
                )

                # Try to fetch embedding
                try:
                    emb_result = session.run(
                        """
                        MATCH (e:Entity {id: $id})
                        RETURN e.embedding AS embedding
                    """,
                        id=entity.id,
                    )
                    emb_record = emb_result.single()
                    if emb_record and emb_record["embedding"]:
                        entity.embedding = emb_record["embedding"]
                except Exception as exc:
                    logger.debug("Could not load entity embedding: %s", exc)

                entities.append(entity)

            return entities

    # -----------------------------------------------------------------
    # COMMUNITY OPERATIONS
    # -----------------------------------------------------------------

    def create_community(self, community: CommunityNode):
        """Create Community node with embedding"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Community {id: $id})
                SET c.corpus_id = $corpus_id,
                    c.level = $level,
                    c.title = $title,
                    c.summary = $summary,
                    c.importance_score = $importance_score,
                    c.entity_count = $entity_count,
                    c.member_hash = $member_hash,
                    c.key_entities = $key_entities,
                    c.key_relationships = $key_relationships,
                    c.created_at = datetime()
            """,
                id=community.id,
                corpus_id=community.corpus_id,
                level=community.level,
                title=community.title,
                summary=community.summary[:5000],
                importance_score=community.importance_score,
                entity_count=community.entity_count,
                member_hash=community.member_hash,
                key_entities=community.key_entities,
                key_relationships=community.key_relationships,
            )

            if community.embedding and any(v != 0 for v in community.embedding):
                session.run(
                    """
                    MATCH (c:Community {id: $id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                    RETURN c
                """,
                    id=community.id,
                    embedding=community.embedding,
                )

    def create_entity_in_community(self, entity_id: str, community_id: str):
        """Create IN_COMMUNITY relationship from Entity to Community"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {id: $entity_id})
                MATCH (c:Community {id: $community_id})
                MERGE (e)-[:IN_COMMUNITY]->(c)
            """,
                entity_id=entity_id,
                community_id=community_id,
            )

    def create_parent_community_rel(self, child_id: str, parent_id: str):
        """Create PARENT_COMMUNITY relationship from child to parent Community"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (child:Community {id: $child_id})
                MATCH (parent:Community {id: $parent_id})
                MERGE (child)-[:PARENT_COMMUNITY]->(parent)
            """,
                child_id=child_id,
                parent_id=parent_id,
            )

    def clear_communities(self, corpus_id: str) -> int:
        """Delete all Community nodes and remove communities property from entities"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Community {corpus_id: $corpus_id})
                DETACH DELETE c
                RETURN count(c) AS deleted
            """,
                corpus_id=corpus_id,
            )
            deleted = result.single()["deleted"]

            # Remove the GDS-written communities property from entities
            session.run(
                """
                MATCH (e:Entity {corpus_id: $corpus_id})
                WHERE e.communities IS NOT NULL
                REMOVE e.communities
            """,
                corpus_id=corpus_id,
            )

            return deleted

    def fetch_existing_community_hashes(self, corpus_id: str) -> Dict[str, Dict]:
        """Fetch existing community member hashes for LLM caching"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Community {corpus_id: $corpus_id})
                WHERE c.member_hash IS NOT NULL AND c.summary IS NOT NULL
                RETURN c.member_hash AS member_hash, c.title AS title,
                       c.summary AS summary, c.importance_score AS importance_score,
                       c.key_entities AS key_entities,
                       c.key_relationships AS key_relationships
            """,
                corpus_id=corpus_id,
            )
            return {
                r["member_hash"]: {
                    "title": r["title"],
                    "summary": r["summary"],
                    "importance_score": r["importance_score"],
                    "key_entities": r["key_entities"] or [],
                    "key_relationships": r["key_relationships"] or [],
                }
                for r in result
            }

    def get_stats(self, corpus_id: str) -> Dict:
        """Get graph statistics"""
        with self.driver.session() as session:
            result = session.run(
                """
                OPTIONAL MATCH (s:Source {corpus_id: $corpus_id})
                WITH count(DISTINCT s) AS sources
                OPTIONAL MATCH (t:Topic {corpus_id: $corpus_id})
                WITH sources, count(DISTINCT t) AS topics
                OPTIONAL MATCH (c:Chunk {corpus_id: $corpus_id})
                WITH sources, topics, count(DISTINCT c) AS chunks
                OPTIONAL MATCH (e:Entity {corpus_id: $corpus_id})
                WITH sources, topics, chunks, count(DISTINCT e) AS entities
                OPTIONAL MATCH (comm:Community {corpus_id: $corpus_id})
                WITH sources, topics, chunks, entities, count(DISTINCT comm) AS communities
                RETURN sources, topics, chunks, entities, communities
            """,
                corpus_id=corpus_id,
            )

            record = result.single()
            return {
                "sources": record["sources"] or 0,
                "topics": record["topics"] or 0,
                "chunks": record["chunks"] or 0,
                "entities": record["entities"] or 0,
                "communities": record["communities"] or 0,
            }
