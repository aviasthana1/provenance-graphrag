from provenance_graphrag.chunker import split_content_into_chunks
from provenance_graphrag.parser import flatten_heading_tree, parse_heading_hierarchy


def test_parser_preserves_heading_hierarchy() -> None:
    markdown = """# Retrieval

Overview.

## Local search

Traverse entities.

### Evidence

Return source chunks.

## Global search

Summarize communities.
"""

    roots = parse_heading_hierarchy(markdown)
    flattened = flatten_heading_tree(roots)

    assert [node.title for node in flattened] == [
        "Retrieval",
        "Local search",
        "Evidence",
        "Global search",
    ]
    assert flattened[1].parent_id == flattened[0].id
    assert flattened[2].parent_id == flattened[1].id


def test_heading_ids_are_stable_for_the_same_document() -> None:
    markdown = "# Stable identifiers\n\nThe same input should produce the same IDs."

    first = flatten_heading_tree(parse_heading_hierarchy(markdown))
    second = flatten_heading_tree(parse_heading_hierarchy(markdown))

    assert [node.id for node in first] == [node.id for node in second]


def test_chunker_bounds_long_paragraphs_and_preserves_provenance() -> None:
    content = "x" * 4_500

    chunks = split_content_into_chunks(
        content,
        topic_id="topic-1",
        source_id="source-1",
        corpus_id="corpus-1",
        heading_path="Root > Detail",
    )

    assert len(chunks) == 3
    assert "".join(chunk.content for chunk in chunks) == content
    assert all(len(chunk.content) <= 2_000 for chunk in chunks)
    assert all(chunk.source_id == "source-1" for chunk in chunks)
    assert all(chunk.heading_path == "Root > Detail" for chunk in chunks)


def test_chunk_ids_are_stable_for_repeat_ingestion() -> None:
    kwargs = {
        "content": "First paragraph.\n\nSecond paragraph.",
        "topic_id": "topic-1",
        "source_id": "source-1",
        "corpus_id": "corpus-1",
        "heading_path": "Root",
    }

    first = split_content_into_chunks(**kwargs)
    second = split_content_into_chunks(**kwargs)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
