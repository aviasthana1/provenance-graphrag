"""
GraphRAG Chatbot Agent

An agentic chatbot that uses GraphRAG retrieval strategies over the
provenance_graphrag Neo4j knowledge graph. The agent decides when and what to
retrieve using tool calling via the OpenRouter Responses API.

Each chatbot instance is scoped to a single corpus.

Search strategies (adapted from Microsoft GraphRAG):
- Local Search: Entity-centric retrieval with relationship traversal
- Global Search: Broad aggregation across entity types and topic hierarchy
- Chunk Search: Direct semantic search on text passages with context expansion
"""

import argparse
import json
import sys
from typing import List, Optional

import requests as http_requests

from .config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENROUTER_API_KEY,
    PROJECT_URL,
)
from .embeddings import embed_single_text
from .neo4j_client import Neo4jClientEnhanced
from .utils import logger

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a knowledgeable assistant powered by a GraphRAG knowledge graph.
You have access to a structured knowledge base containing entities (people, organizations,
concepts, technologies), their relationships, source text documents, and community reports
(thematic clusters of related entities with pre-computed summaries).

Your retrieval tools:
- local_search: Find specific entities and their connections. Use when the user asks about
  particular people, concepts, technologies, or relationships between things.
- global_search: Get a broad overview of the knowledge base including community summaries.
  Use when the user asks "what is this about?", "summarize everything", or broad thematic questions.
- chunk_search: Find specific text passages. Use when the user needs exact wording,
  detailed explanations, or step-by-step content from the source documents.
- community_search: Find thematic communities and their summaries. Use when the user asks
  about themes, topic clusters, groups, or how things are connected at a high level.

Guidelines:
1. ALWAYS use at least one search tool before answering questions about the knowledge base.
2. For complex questions, you may call multiple tools (e.g., local_search for entities +
   chunk_search for detailed text).
3. If a search returns no results, try a different tool or rephrase the query.
4. Cite the source information in your answers (entity names, topic paths, etc.).
5. Be honest when information is not found -- do not hallucinate.
6. For follow-up questions, use the conversation context to inform your searches."""


# =============================================================================
# TOOL DEFINITIONS (OpenRouter Responses API format)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "local_search",
        "description": (
            "Search for specific entities, people, concepts, or technologies in the knowledge graph. "
            "This performs entity-centric retrieval: finds the most relevant entities matching the query, "
            "then gathers their relationships, connected entities, and source text chunks for rich context. "
            "Best for questions about specific topics, people, organizations, or how things relate to each other."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what entities or concepts to find",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "global_search",
        "description": (
            "Get a broad overview of all knowledge in the knowledge base. Aggregates information across "
            "all entities, topics, and their relationships to provide high-level summaries. Best for "
            "questions like 'what is this about?', 'summarize the main topics', 'what are the key themes?', "
            "or any question that requires understanding the full scope of the knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The broad question or topic to explore across the knowledge base",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "chunk_search",
        "description": (
            "Search for specific text passages and content in the knowledge base using semantic similarity. "
            "Retrieves the most relevant text chunks along with their surrounding context (neighboring chunks). "
            "Best for finding exact information, quotes, detailed explanations, step-by-step procedures, "
            "or when you need the original source text rather than entity summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what content to find",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "community_search",
        "description": (
            "Search for thematic communities in the knowledge graph. Communities are automatically "
            "detected clusters of related entities with pre-computed summaries. Use this when the user "
            "asks about themes, topic clusters, groups, or wants to understand how groups of things "
            "are connected at a high level."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for relevant communities",
                }
            },
            "required": ["query"],
        },
    },
]


# =============================================================================
# GRAPHRAG RETRIEVER
# =============================================================================


class GraphRAGRetriever:
    """Retrieval engine for GraphRAG search, scoped to a single corpus."""

    def __init__(self, neo4j_driver, corpus_id: str):
        self.driver = neo4j_driver
        self.corpus_id = corpus_id

    def _run_cypher(self, query: str, params: dict = None) -> list:
        """Execute Cypher query and return list of record dicts."""
        with self.driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def _embed_query(self, text: str) -> List[float]:
        """Embed query text using Gemini with RETRIEVAL_QUERY task type."""
        return embed_single_text(text, task_type="RETRIEVAL_QUERY")

    # -------------------------------------------------------------------------
    # LOCAL SEARCH — Entity-centric retrieval
    # -------------------------------------------------------------------------

    def local_search(self, query: str, top_k: int = 5) -> str:
        """
        Entity-centric search: embed query -> vector search entities ->
        traverse graph for relationships + chunks -> format context.
        """
        query_embedding = self._embed_query(query)

        # Step 1: Vector search on entities (oversample 3x for corpus filtering)
        entities = self._run_cypher(
            """
            CALL db.index.vector.queryNodes('entity_embedding', $oversample, $embedding)
            YIELD node AS e, score
            WHERE e.corpus_id = $corpus_id
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.description AS description, score
            ORDER BY score DESC
            LIMIT $top_k
        """,
            {
                "embedding": query_embedding,
                "corpus_id": self.corpus_id,
                "oversample": top_k * 3,
                "top_k": top_k,
            },
        )

        # Fallback: full-text search if vector returns few results
        if len(entities) < 2:
            ft_entities = self._run_cypher(
                """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $search_text)
                YIELD node, score
                WHERE node.corpus_id = $corpus_id
                RETURN node.id AS id, node.name AS name, node.type AS type,
                       node.description AS description, score
                LIMIT $top_k
            """,
                {"search_text": query, "corpus_id": self.corpus_id, "top_k": top_k},
            )
            existing_ids = {e["id"] for e in entities}
            for e in ft_entities:
                if e["id"] not in existing_ids:
                    entities.append(e)

        if not entities:
            return f"No entities found matching '{query}'."

        entity_ids = [e["id"] for e in entities]

        # Step 2: Get relationships for matched entities
        relationships = self._run_cypher(
            """
            MATCH (e:Entity)-[r]->(other:Entity)
            WHERE e.id IN $entity_ids AND type(r) <> 'SIMILAR'
            RETURN e.name AS from_name, type(r) AS rel_type,
                   other.name AS to_name, other.type AS to_type,
                   other.description AS to_description,
                   r.confidence AS confidence
        """,
            {"entity_ids": entity_ids},
        )

        # Also get incoming relationships
        incoming = self._run_cypher(
            """
            MATCH (other:Entity)-[r]->(e:Entity)
            WHERE e.id IN $entity_ids AND type(r) <> 'SIMILAR'
            AND NOT other.id IN $entity_ids
            RETURN other.name AS from_name, type(r) AS rel_type,
                   e.name AS to_name, e.type AS to_type,
                   e.description AS to_description,
                   r.confidence AS confidence
        """,
            {"entity_ids": entity_ids},
        )
        relationships.extend(incoming)

        # Step 3: Get source chunks that mention matched entities
        chunks = self._run_cypher(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.id IN $entity_ids
            OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(t:Topic)
            WITH c, t, collect(DISTINCT e.name) AS mentioned_entities
            ORDER BY c.chunk_index
            LIMIT $limit
            RETURN c.id AS chunk_id, c.content AS content,
                   c.heading_path AS heading_path, t.title AS topic_title,
                   mentioned_entities
        """,
            {"entity_ids": entity_ids, "limit": top_k * 2},
        )

        return self._format_entity_context(entities, relationships, chunks)

    # -------------------------------------------------------------------------
    # GLOBAL SEARCH — Broad aggregation
    # -------------------------------------------------------------------------

    def global_search(self, query: str) -> str:
        """
        Broad aggregation search: entity types, topic hierarchy, hub entities,
        key relationships.
        """
        nb = self.corpus_id

        # Entity type breakdown
        entity_types = self._run_cypher(
            """
            MATCH (e:Entity {corpus_id: $corpus_id})
            RETURN e.type AS entity_type, count(e) AS count,
                   collect(e.name)[..10] AS sample_names
            ORDER BY count DESC
        """,
            {"corpus_id": nb},
        )

        # Topic hierarchy
        topics = self._run_cypher(
            """
            MATCH (t:Topic {corpus_id: $corpus_id})
            OPTIONAL MATCH (t)-[:HAS_SUBTOPIC]->(sub:Topic)
            WITH t, collect(sub.title) AS subtopics
            ORDER BY t.level, t.position
            RETURN t.id AS id, t.title AS title, t.level AS level,
                   t.summary AS summary, subtopics
        """,
            {"corpus_id": nb},
        )

        # Hub entities (most connected)
        hub_entities = self._run_cypher(
            """
            MATCH (e:Entity {corpus_id: $corpus_id})-[r]-(other)
            WHERE type(r) <> 'SIMILAR'
            WITH e, count(r) AS rel_count
            ORDER BY rel_count DESC
            LIMIT 15
            RETURN e.name AS name, e.type AS type,
                   e.description AS description, rel_count
        """,
            {"corpus_id": nb},
        )

        # Key relationships
        key_relationships = self._run_cypher(
            """
            MATCH (e1:Entity {corpus_id: $corpus_id})-[r]->(e2:Entity)
            WHERE type(r) IN ['RELATED_TO', 'PART_OF', 'USES', 'CREATED_BY',
                              'WORKS_AT', 'STUDIED_AT', 'HAS_SKILL', 'ENABLES']
            AND (r.confidence IS NULL OR r.confidence >= 0.7)
            RETURN e1.name AS from_name, type(r) AS relationship,
                   e2.name AS to_name
            ORDER BY r.confidence DESC
            LIMIT 30
        """,
            {"corpus_id": nb},
        )

        # Source documents
        sources = self._run_cypher(
            """
            MATCH (s:Source {corpus_id: $corpus_id})
            RETURN s.title AS title, s.source_type AS source_type
        """,
            {"corpus_id": nb},
        )

        # Community reports
        community_reports = self._run_cypher(
            """
            MATCH (c:Community {corpus_id: $corpus_id})
            WHERE c.summary IS NOT NULL
            RETURN c.title AS title, c.summary AS summary, c.level AS level,
                   c.importance_score AS importance_score,
                   c.entity_count AS entity_count,
                   c.key_entities AS key_entities
            ORDER BY c.importance_score DESC
            LIMIT 10
        """,
            {"corpus_id": nb},
        )

        return self._format_global_context(
            entity_types, topics, hub_entities, key_relationships, sources, community_reports
        )

    # -------------------------------------------------------------------------
    # CHUNK SEARCH — Direct content retrieval
    # -------------------------------------------------------------------------

    def chunk_search(self, query: str, top_k: int = 5) -> str:
        """
        Direct content search: embed query -> vector search on chunks ->
        include neighboring chunks for context window.
        """
        query_embedding = self._embed_query(query)

        # Vector search on chunks
        chunks = self._run_cypher(
            """
            CALL db.index.vector.queryNodes('chunk_embedding', $oversample, $embedding)
            YIELD node AS c, score
            WHERE c.corpus_id = $corpus_id
            RETURN c.id AS id, c.content AS content, c.heading_path AS heading_path,
                   c.chunk_index AS chunk_index, score
            ORDER BY score DESC
            LIMIT $top_k
        """,
            {
                "embedding": query_embedding,
                "corpus_id": self.corpus_id,
                "oversample": top_k * 3,
                "top_k": top_k,
            },
        )

        # Fallback: full-text search
        if len(chunks) < 2:
            ft_chunks = self._run_cypher(
                """
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $search_text)
                YIELD node, score
                WHERE node.corpus_id = $corpus_id
                RETURN node.id AS id, node.content AS content,
                       node.heading_path AS heading_path,
                       node.chunk_index AS chunk_index, score
                LIMIT $top_k
            """,
                {"search_text": query, "corpus_id": self.corpus_id, "top_k": top_k},
            )
            existing_ids = {c["id"] for c in chunks}
            for c in ft_chunks:
                if c["id"] not in existing_ids:
                    chunks.append(c)

        if not chunks:
            return f"No text passages found matching '{query}'."

        # Get neighboring chunks and entities for each result
        enriched_chunks = []
        for chunk in chunks:
            # Neighbors
            neighbors = self._run_cypher(
                """
                MATCH (c:Chunk {id: $id})
                OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
                OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
                RETURN prev.content AS prev_content,
                       next.content AS next_content
            """,
                {"id": chunk["id"]},
            )

            # Entities mentioned in this chunk
            mentioned = self._run_cypher(
                """
                MATCH (c:Chunk {id: $id})-[:MENTIONS]->(e:Entity)
                RETURN e.name AS name, e.type AS type
            """,
                {"id": chunk["id"]},
            )

            enriched = dict(chunk)
            if neighbors:
                enriched["prev_content"] = neighbors[0].get("prev_content")
                enriched["next_content"] = neighbors[0].get("next_content")
            enriched["entities"] = mentioned
            enriched_chunks.append(enriched)

        return self._format_chunk_context(enriched_chunks)

    # -------------------------------------------------------------------------
    # COMMUNITY SEARCH — Thematic cluster retrieval
    # -------------------------------------------------------------------------

    def community_search(self, query: str, top_k: int = 5) -> str:
        """
        Search communities by embedding similarity + fulltext fallback.
        """
        query_embedding = self._embed_query(query)

        # Vector search on community embeddings
        communities = self._run_cypher(
            """
            CALL db.index.vector.queryNodes('community_embedding', $oversample, $embedding)
            YIELD node AS c, score
            WHERE c.corpus_id = $corpus_id
            RETURN c.title AS title, c.summary AS summary, c.level AS level,
                   c.importance_score AS importance_score,
                   c.key_entities AS key_entities,
                   c.key_relationships AS key_relationships,
                   c.entity_count AS entity_count, score
            ORDER BY score DESC
            LIMIT $top_k
        """,
            {
                "embedding": query_embedding,
                "corpus_id": self.corpus_id,
                "oversample": top_k * 3,
                "top_k": top_k,
            },
        )

        # Fallback: full-text search
        if len(communities) < 2:
            ft_communities = self._run_cypher(
                """
                CALL db.index.fulltext.queryNodes('community_fulltext', $search_text)
                YIELD node, score
                WHERE node.corpus_id = $corpus_id
                RETURN node.title AS title, node.summary AS summary,
                       node.level AS level, node.importance_score AS importance_score,
                       node.key_entities AS key_entities,
                       node.key_relationships AS key_relationships,
                       node.entity_count AS entity_count, score
                LIMIT $top_k
            """,
                {"search_text": query, "corpus_id": self.corpus_id, "top_k": top_k},
            )
            existing_titles = {c["title"] for c in communities}
            for c in ft_communities:
                if c["title"] not in existing_titles:
                    communities.append(c)

        if not communities:
            return (
                "No communities found in this corpus. Communities may not have been generated yet."
            )

        return self._format_community_context(communities)

    # -------------------------------------------------------------------------
    # CONTEXT FORMATTERS
    # -------------------------------------------------------------------------

    def _format_entity_context(self, entities, relationships, chunks) -> str:
        sections = []

        if entities:
            entity_lines = []
            for e in entities:
                score = f" [relevance: {e['score']:.2f}]" if e.get("score") else ""
                desc = e.get("description") or "No description"
                entity_lines.append(f"- {e['name']} ({e['type']}): {desc[:200]}{score}")
            sections.append("=== MATCHED ENTITIES ===\n" + "\n".join(entity_lines))

        if relationships:
            rel_lines = []
            seen = set()
            for r in relationships:
                key = (r["from_name"], r["rel_type"], r["to_name"])
                if key in seen:
                    continue
                seen.add(key)
                desc = ""
                if r.get("to_description"):
                    desc = f" ({r['to_description'][:100]})"
                rel_lines.append(f"- {r['from_name']} --[{r['rel_type']}]--> {r['to_name']}{desc}")
            sections.append("=== RELATIONSHIPS ===\n" + "\n".join(rel_lines[:25]))

        if chunks:
            chunk_lines = []
            for c in chunks:
                mentioned = ", ".join(c.get("mentioned_entities", []))
                heading = c.get("heading_path") or "Unknown section"
                chunk_lines.append(f"--- [{heading}] ---\n{c['content']}\n(Mentions: {mentioned})")
            sections.append("=== SOURCE TEXT ===\n" + "\n\n".join(chunk_lines[:8]))

        if not sections:
            return "No relevant entities found in the knowledge graph."

        return "\n\n".join(sections)

    def _format_community_context(self, communities) -> str:
        if not communities:
            return "No communities found in this corpus."

        sections = []
        for i, c in enumerate(communities, 1):
            parts = [f"=== COMMUNITY {i}: {c['title']} (Level {c.get('level', 0)}) ==="]
            parts.append(f"Summary: {c['summary']}")
            parts.append(f"Importance: {c.get('importance_score', 'N/A')}/10")
            parts.append(f"Entities: {c.get('entity_count', '?')}")

            key_entities = c.get("key_entities")
            if key_entities:
                if isinstance(key_entities, str):
                    try:
                        key_entities = json.loads(key_entities)
                    except (json.JSONDecodeError, TypeError):
                        key_entities = []
                if key_entities:
                    parts.append(f"Key entities: {', '.join(str(e) for e in key_entities[:8])}")

            key_rels = c.get("key_relationships")
            if key_rels:
                if isinstance(key_rels, str):
                    try:
                        key_rels = json.loads(key_rels)
                    except (json.JSONDecodeError, TypeError):
                        key_rels = []
                if key_rels:
                    parts.append(f"Key relationships: {'; '.join(str(r) for r in key_rels[:5])}")

            sections.append("\n".join(parts))

        return "\n\n".join(sections)

    def _format_global_context(
        self, entity_types, topics, hub_entities, key_relationships, sources, community_reports=None
    ) -> str:
        sections = []

        if sources:
            src_lines = [f"- {s['title']}" for s in sources]
            sections.append("=== SOURCE DOCUMENTS ===\n" + "\n".join(src_lines))

        if community_reports:
            comm_lines = []
            for c in community_reports:
                importance = c.get("importance_score", "?")
                entity_count = c.get("entity_count", "?")
                level_str = f" (L{c.get('level', 0)})" if c.get("level", 0) > 0 else ""
                comm_lines.append(
                    f"- {c['title']}{level_str} [{importance}/10, {entity_count} entities]: "
                    f"{c['summary'][:200]}"
                )
            sections.append("=== COMMUNITY OVERVIEW ===\n" + "\n".join(comm_lines))

        if topics:
            topic_lines = []
            for t in topics:
                indent = "  " * max(0, (t.get("level") or 1) - 1)
                subtopics = ", ".join(t.get("subtopics", [])[:5])
                summary = f": {t['summary'][:150]}" if t.get("summary") else ""
                sub_info = f" (subtopics: {subtopics})" if subtopics else ""
                topic_lines.append(f"{indent}- {t['title']}{summary}{sub_info}")
            sections.append("=== DOCUMENT STRUCTURE ===\n" + "\n".join(topic_lines))

        if entity_types:
            type_lines = []
            for et in entity_types:
                samples = ", ".join(et["sample_names"][:7])
                type_lines.append(f"- {et['entity_type']} ({et['count']} entities): {samples}")
            sections.append("=== ENTITY TYPES ===\n" + "\n".join(type_lines))

        if hub_entities:
            hub_lines = []
            for h in hub_entities:
                desc = h.get("description") or "No description"
                hub_lines.append(
                    f"- {h['name']} ({h['type']}): {desc[:150]} [{h['rel_count']} connections]"
                )
            sections.append("=== KEY ENTITIES (most connected) ===\n" + "\n".join(hub_lines))

        if key_relationships:
            rel_lines = []
            for r in key_relationships:
                rel_lines.append(f"- {r['from_name']} --[{r['relationship']}]--> {r['to_name']}")
            sections.append("=== KEY RELATIONSHIPS ===\n" + "\n".join(rel_lines[:25]))

        if not sections:
            return "No data found in this corpus."

        return "\n\n".join(sections)

    def _format_chunk_context(self, enriched_chunks) -> str:
        if not enriched_chunks:
            return "No relevant text passages found."

        sections = []
        for i, chunk in enumerate(enriched_chunks, 1):
            parts = [f"=== RESULT {i} (relevance: {chunk.get('score', 0):.2f}) ==="]
            parts.append(f"Section: {chunk.get('heading_path', 'Unknown')}")

            if chunk.get("prev_content"):
                parts.append(f"[Previous context]: {chunk['prev_content'][:300]}...")

            parts.append(f"[Main passage]: {chunk['content']}")

            if chunk.get("next_content"):
                parts.append(f"[Following context]: {chunk['next_content'][:300]}...")

            if chunk.get("entities"):
                entity_names = [f"{e['name']} ({e['type']})" for e in chunk["entities"]]
                parts.append(f"Entities mentioned: {', '.join(entity_names)}")

            sections.append("\n".join(parts))

        return "\n\n".join(sections)


# =============================================================================
# CHAT AGENT
# =============================================================================

OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"


class ChatAgent:
    """Agentic chatbot scoped to a single corpus, using OpenRouter Responses API."""

    def __init__(self, retriever: GraphRAGRetriever, model: str = "google/gemini-3-flash-preview"):
        self.retriever = retriever
        self.model = model
        self.conversation_history: List[dict] = []
        self.tools = TOOL_DEFINITIONS

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def chat(self, user_message: str) -> tuple:
        """Process user message through the agentic loop.
        Returns (assistant_response, usage_stats) tuple."""
        self.conversation_history.append(
            {"type": "message", "role": "user", "content": user_message}
        )
        return self._agentic_loop()

    def _agentic_loop(self) -> tuple:
        """
        Core loop:
        1. Send input + tools to LLM via Responses API
        2. If response has function_call items, execute them
        3. Append function_call + function_call_output to input
        4. Call LLM again
        5. Repeat until LLM returns a message (no function calls)

        Returns (assistant_response, usage_stats) tuple.
        usage_stats is a dict with input_tokens, output_tokens, total_tokens
        accumulated across all LLM calls in this turn.
        """
        max_iterations = 5
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for iteration in range(max_iterations):
            response_data = self._call_llm()

            if not response_data or "output" not in response_data:
                error_msg = (
                    response_data.get("error", {}).get("message", "Unknown error")
                    if response_data
                    else "No response"
                )
                logger.error(f"LLM call failed: {error_msg}")
                return f"Sorry, I encountered an error: {error_msg}", total_usage

            # Accumulate usage from this call
            usage = response_data.get("usage", {})
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)

            output_items = response_data["output"]

            # Separate function calls and messages
            function_calls = [item for item in output_items if item.get("type") == "function_call"]
            messages = [item for item in output_items if item.get("type") == "message"]

            if function_calls:
                # Add output items (including any function_call items) to history
                for item in output_items:
                    self.conversation_history.append(item)

                # Execute each function call and add results
                for fc in function_calls:
                    tool_name = fc["name"]
                    try:
                        arguments = json.loads(fc["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}

                    logger.info(f"  Tool call: {tool_name}({json.dumps(arguments)[:100]})")
                    result = self._execute_tool(tool_name, arguments)

                    self.conversation_history.append(
                        {"type": "function_call_output", "call_id": fc["call_id"], "output": result}
                    )

                # Continue loop for LLM to process tool results
                continue

            # No function calls — extract final text response
            if messages:
                msg = messages[0]
                content = msg.get("content", [])
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)

                assistant_response = "\n".join(text_parts) if text_parts else str(content)
                self.conversation_history.append(
                    {"type": "message", "role": "assistant", "content": assistant_response}
                )
                return assistant_response, total_usage

            # Fallback: try to extract any text from output
            return "I received an unexpected response format. Please try again.", total_usage

        return (
            "I reached the maximum number of retrieval steps. Please try a simpler question.",
            total_usage,
        )

    def _call_llm(self) -> Optional[dict]:
        """Make OpenRouter Responses API call."""
        payload = {
            "model": self.model,
            "input": self.conversation_history,
            "instructions": SYSTEM_PROMPT,
            "tools": self.tools,
            "temperature": 0.3,
            "max_output_tokens": 4096,
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": PROJECT_URL,
            "X-Title": "GraphRAG Chatbot",
        }

        try:
            response = http_requests.post(
                OPENROUTER_RESPONSES_URL, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()
        except http_requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            try:
                error_body = e.response.json()
                return {"error": {"message": error_body.get("error", {}).get("message", str(e))}}
            except Exception:
                return {"error": {"message": str(e)}}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": {"message": str(e)}}

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Route tool call to the appropriate retriever method."""
        try:
            if tool_name == "local_search":
                return self.retriever.local_search(query=arguments["query"])
            elif tool_name == "global_search":
                return self.retriever.global_search(query=arguments["query"])
            elif tool_name == "chunk_search":
                return self.retriever.chunk_search(query=arguments["query"])
            elif tool_name == "community_search":
                return self.retriever.community_search(query=arguments["query"])
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return f"Error executing {tool_name}: {str(e)}"


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """CLI entry point for the GraphRAG chatbot."""
    parser = argparse.ArgumentParser(description="GraphRAG Chatbot - Query your knowledge graph")
    parser.add_argument("corpus_id", help="Corpus ID to query")
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model to use (default: google/gemini-3-flash-preview)",
    )
    args = parser.parse_args()

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    client = Neo4jClientEnhanced(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if not client.test_connection():
        print("Failed to connect to Neo4j. Check your credentials.")
        sys.exit(1)
    print(f"Connected. Corpus: {args.corpus_id}")

    # Create retriever and agent
    retriever = GraphRAGRetriever(client.driver, corpus_id=args.corpus_id)
    agent = ChatAgent(retriever, model=args.model)

    print("\nGraphRAG Chatbot ready.\nCommands: /clear, /quit\n---")

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input == "/quit":
                break
            elif user_input == "/clear":
                agent.reset_conversation()
                print("Conversation cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)
            response, usage = agent.chat(user_input)
            print(response)
            print(
                f"\n[Tokens: in={usage['input_tokens']}, "
                f"out={usage['output_tokens']}, "
                f"total={usage['total_tokens']}]"
            )

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        client.close()


if __name__ == "__main__":
    main()
