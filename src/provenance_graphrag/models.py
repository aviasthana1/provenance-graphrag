"""
Data models for the Provenance GraphRAG
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HeadingNode:
    """A heading in the document hierarchy"""

    id: str
    title: str
    level: int
    content: str
    children: List["HeadingNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    position: int = 0


@dataclass
class TopicNode:
    """Topic node (corresponds to a heading)"""

    id: str
    corpus_id: str
    title: str
    level: int
    summary: str
    content: str
    embedding: List[float] = field(default_factory=list)
    parent_id: Optional[str] = None
    position: int = 0


@dataclass
class ChunkNode:
    """Chunk node - content piece"""

    id: str
    source_id: str
    corpus_id: str
    topic_id: str
    content: str
    chunk_index: int
    heading_path: str
    embedding: List[float] = field(default_factory=list)


@dataclass
class EntityNode:
    """Entity extracted from chunks"""

    id: str
    name: str
    type: str  # PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, METHOD
    description: str
    source_chunks: List[str] = field(default_factory=list)  # Chunk IDs where mentioned
    embedding: List[float] = field(default_factory=list)
    normalized_name: str = ""  # For entity resolution


@dataclass
class RelationshipEdge:
    """Relationship between entities or topics"""

    from_id: str
    to_id: str
    type: str  # RELATED_TO, PREREQUISITE_OF, SIMILAR, PART_OF, etc.
    confidence: float = 1.0
    inferred: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunityNode:
    """A detected community of related entities."""

    id: str
    corpus_id: str
    level: int
    title: str
    summary: str
    key_entities: List[str] = field(default_factory=list)
    key_relationships: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    entity_count: int = 0
    parent_community_id: Optional[str] = None
    member_entity_ids: List[str] = field(default_factory=list)
    member_hash: str = ""
    embedding: List[float] = field(default_factory=list)
