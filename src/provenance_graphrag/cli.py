#!/usr/bin/env python3
"""
Command-line interface for the Provenance GraphRAG
"""

import argparse
import json
import os
import uuid

from .pipeline import build_enhanced_knowledge_graph
from .utils import logger


def main():
    parser = argparse.ArgumentParser(
        description="Provenance GraphRAG with Entity Extraction & Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build new graph from first document
    python -m provenance_graphrag --text-file doc1.md --corpus-id abc123

    # Generate corpus ID automatically
    python -m provenance_graphrag --text-file doc1.md --generate-corpus-id

Features:
    - Batch entity extraction with Mistral Nemo (5 chunks per call)
    - Entity resolution with fuzzy matching + embeddings
    - KNN similarity graphs (Topics, Entities, Chunks)
    - Transitive & lexical inference
    - Full-text indexes for hybrid search
        """,
    )

    parser.add_argument("--text-file", help="Path to text/markdown file")
    parser.add_argument("--text", help="Inline text content")
    parser.add_argument("--corpus-id", help="Corpus ID")
    parser.add_argument("--generate-corpus-id", action="store_true", help="Auto-generate corpus ID")
    parser.add_argument("--source-title", default="Document", help="Source document title")

    args = parser.parse_args()

    if not args.text_file and not args.text:
        parser.error("Either --text-file or --text is required")

    if args.generate_corpus_id:
        corpus_id = str(uuid.uuid4())
        logger.info(f"Generated corpus ID: {corpus_id}")
    elif args.corpus_id:
        corpus_id = args.corpus_id
    else:
        parser.error("Either --corpus-id or --generate-corpus-id is required")

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Read {len(text)} characters from {args.text_file}")
        if args.source_title == "Document":
            args.source_title = (
                os.path.basename(args.text_file).replace(".md", "").replace("_", " ").title()
            )
    else:
        text = args.text.replace("\\n", "\n")

    # Build the enhanced graph
    result = build_enhanced_knowledge_graph(text, corpus_id, args.source_title)

    print("\n" + "=" * 70)
    print("RESULT JSON:")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
