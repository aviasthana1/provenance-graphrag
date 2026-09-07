import pytest

from provenance_graphrag.inference import apply_transitive_inference
from provenance_graphrag.models import RelationshipEdge
from provenance_graphrag.neo4j_client import validate_relationship_type
from provenance_graphrag.similarity import build_knn_similarity_graph


def test_knn_respects_neighbor_limit_and_threshold() -> None:
    nodes = [
        ("a", [1.0, 0.0], "Entity"),
        ("b", [0.99, 0.01], "Entity"),
        ("c", [0.0, 1.0], "Entity"),
    ]

    edges = build_knn_similarity_graph(nodes, k=1, min_score=0.9)

    assert {(edge.from_id, edge.to_id) for edge in edges} == {("a", "b"), ("b", "a")}
    assert all(edge.type == "SIMILAR" for edge in edges)


def test_transitive_inference_adds_one_grounded_edge() -> None:
    direct = [
        RelationshipEdge("a", "b", "DEPENDS_ON"),
        RelationshipEdge("b", "c", "DEPENDS_ON"),
    ]

    inferred = apply_transitive_inference(direct, {"b": "middle"})

    assert len(inferred) == 1
    assert inferred[0].from_id == "a"
    assert inferred[0].to_id == "c"
    assert inferred[0].properties["via"] == "b"
    assert inferred[0].inferred is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [("related to", "RELATED_TO"), ("part-of", "PART_OF"), ("USES", "USES")],
)
def test_relationship_types_are_normalized(value: str, expected: str) -> None:
    assert validate_relationship_type(value) == expected


@pytest.mark.parametrize("value", ["", "x`]->(n) DELETE n //", "123", "A/B"])
def test_relationship_types_reject_cypher_fragments(value: str) -> None:
    with pytest.raises(ValueError):
        validate_relationship_type(value)
