"""
Document parsing - extracts heading hierarchy from markdown
"""

import re
import uuid
from typing import List

from .models import HeadingNode


def parse_heading_hierarchy(text: str) -> List[HeadingNode]:
    """
    Parse markdown text and extract heading hierarchy.
    Returns a list of root-level headings (forest structure).
    """
    lines = text.split("\n")
    headings_raw = []
    current_content_lines = []
    current_heading = None

    for i, line in enumerate(lines):
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            if current_heading is not None:
                current_heading["content"] = "\n".join(current_content_lines).strip()

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            current_heading = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"heading:{i}:{level}:{title}")),
                "title": title,
                "level": level,
                "content": "",
                "line": i,
                "position": len(headings_raw),
            }
            headings_raw.append(current_heading)
            current_content_lines = []
        else:
            current_content_lines.append(line)

    if current_heading is not None:
        current_heading["content"] = "\n".join(current_content_lines).strip()

    # Handle case where there's content before any heading
    if not headings_raw and text.strip():
        headings_raw.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "heading:0:1:Document")),
                "title": "Document",
                "level": 1,
                "content": text.strip(),
                "line": 0,
                "position": 0,
            }
        )

    # Convert to HeadingNode objects
    heading_nodes = []
    for h in headings_raw:
        heading_nodes.append(
            HeadingNode(
                id=h["id"],
                title=h["title"],
                level=h["level"],
                content=h["content"],
                children=[],
                parent_id=None,
                position=h["position"],
            )
        )

    # Build parent-child relationships based on heading levels
    root_nodes = []
    stack: List[HeadingNode] = []

    for node in heading_nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            parent = stack[-1]
            node.parent_id = parent.id
            parent.children.append(node)
        else:
            root_nodes.append(node)

        stack.append(node)

    return root_nodes


def flatten_heading_tree(roots: List[HeadingNode]) -> List[HeadingNode]:
    """Flatten heading tree into list (depth-first order)"""
    result = []

    def traverse(node: HeadingNode):
        result.append(node)
        for child in node.children:
            traverse(child)

    for root in roots:
        traverse(root)

    return result
