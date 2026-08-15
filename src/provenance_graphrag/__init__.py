"""Build and query provenance-aware knowledge graphs."""

from .chunker import split_content_into_chunks
from .entity_resolution import resolve_entities_with_existing
from .inference import (
    apply_lexical_similarity_inference,
    apply_transitive_inference,
)
from .parser import flatten_heading_tree, parse_heading_hierarchy
from .similarity import build_knn_similarity_graph

__all__ = [
    "apply_lexical_similarity_inference",
    "apply_transitive_inference",
    "build_knn_similarity_graph",
    "flatten_heading_tree",
    "parse_heading_hierarchy",
    "resolve_entities_with_existing",
    "split_content_into_chunks",
]

__version__ = "0.1.0"
