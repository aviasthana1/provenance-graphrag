"""
Content chunking - splits content into smaller pieces
"""

import re
import uuid
from typing import List

from .config import MAX_CHUNK_CHARS
from .models import ChunkNode


def _split_oversized_paragraph(paragraph: str) -> List[str]:
    """Split a paragraph into bounded windows without dropping text."""
    return [
        paragraph[start : start + MAX_CHUNK_CHARS]
        for start in range(0, len(paragraph), MAX_CHUNK_CHARS)
    ]


def split_content_into_chunks(
    content: str,
    topic_id: str,
    source_id: str,
    corpus_id: str,
    heading_path: str,
    start_index: int = 0,
) -> List[ChunkNode]:
    """
    Split content into chunks for a topic.
    Splits by paragraphs, respecting MAX_CHUNK_CHARS limit.
    """
    if not content.strip():
        return []

    chunks = []
    paragraphs = [
        part
        for paragraph in re.split(r"\n\s*\n", content)
        for part in _split_oversized_paragraph(paragraph.strip())
        if part
    ]

    current_chunk = []
    current_length = 0
    chunk_index = start_index

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_length + len(para) > MAX_CHUNK_CHARS and current_chunk:
            # Save current chunk
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(
                ChunkNode(
                    id=str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{chunk_index}:{chunk_content}")
                    ),
                    source_id=source_id,
                    corpus_id=corpus_id,
                    topic_id=topic_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    heading_path=heading_path,
                )
            )
            chunk_index += 1
            current_chunk = []
            current_length = 0

        current_chunk.append(para)
        current_length += len(para)

    # Save remaining content
    if current_chunk:
        chunk_content = "\n\n".join(current_chunk)
        chunks.append(
            ChunkNode(
                id=str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{chunk_index}:{chunk_content}")
                ),
                source_id=source_id,
                corpus_id=corpus_id,
                topic_id=topic_id,
                content=chunk_content,
                chunk_index=chunk_index,
                heading_path=heading_path,
            )
        )

    return chunks
