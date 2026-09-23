from evalharness.metrics.retrieval import hit_at_k, reciprocal_rank


def test_hit_when_expected_in_top_k():
    assert hit_at_k(["a", "b", "c"], ["c"], k=3)


def test_miss_when_expected_outside_top_k():
    assert not hit_at_k(["a", "b", "c"], ["c"], k=2)


def test_any_expected_source_counts():
    assert hit_at_k(["a", "b"], ["x", "b"], k=2)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["b"], k=3) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["z"], k=3) == 0.0
