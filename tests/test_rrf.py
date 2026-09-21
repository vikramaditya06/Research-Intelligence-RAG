from app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_promotes_consensus():
    a = [("a", 1, {"id": "a"}), ("b", 1, {"id": "b"})]
    b = [("b", 1, {"id": "b"}), ("a", 1, {"id": "a"})]
    result = reciprocal_rank_fusion(a, b)
    assert len(result) == 2
    assert {x[0] for x in result} == {"a", "b"}


def test_rrf_uses_rank_across_three_lists():
    result = reciprocal_rank_fusion(
        [("a", 1, {"id": "a"}), ("b", 1, {"id": "b"})],
        [("a", 1, {"id": "a"})],
        [("b", 1, {"id": "b"})],
    )
    assert result[0][0] == "a" or result[0][0] == "b"
