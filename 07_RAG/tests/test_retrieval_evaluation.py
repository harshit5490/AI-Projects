
import pytest

from app.evaluation.retrieval_metrics import hit_at_k,reciprocal_rank,mean_reciprocal_rank,mean_hit_at_k


def test_hit_at_k_when_relevant_chunk_is_first():
    assert hit_at_k(
        ["chunk-1", "chunk-2"],
        {"chunk-1"},
        1,
    ) == 1.0


def test_hit_at_k_when_relevant_chunk_is_within_k():
    assert hit_at_k(
        ["chunk-1", "chunk-2", "chunk-3"],
        {"chunk-3"},
        3,
    ) == 1.0


def test_hit_at_k_when_relevant_chunk_is_outside_k():
    assert hit_at_k(
        ["chunk-1", "chunk-2", "chunk-3"],
        {"chunk-3"},
        2,
    ) == 0.0


def test_hit_at_k_when_no_relevant_chunk_is_retrieved():
    assert hit_at_k(
        ["chunk-1", "chunk-2"],
        {"chunk-3"},
        2,
    ) == 0.0


def test_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be greater than 0"):
        hit_at_k(["chunk-1"], {"chunk-1"}, 0)


def test_hit_at_k_rejects_empty_relevant_set():
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids cannot be empty",
    ):
        hit_at_k(["chunk-1"], set(), 1)


def test_reciprocal_rank_when_relevant_chunk_is_first():
    assert reciprocal_rank(
        ["chunk-1", "chunk-2"],
        {"chunk-1"},
    ) == 1.0


def test_reciprocal_rank_when_relevant_chunk_is_third():
    assert reciprocal_rank(
        ["chunk-1", "chunk-2", "chunk-3"],
        {"chunk-3"},
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_when_no_relevant_chunk_is_found():
    assert reciprocal_rank(
        ["chunk-1", "chunk-2"],
        {"chunk-3"},
    ) == 0.0


def test_reciprocal_rank_uses_first_relevant_chunk():
    assert reciprocal_rank(
        ["chunk-1", "chunk-2", "chunk-3"],
        {"chunk-2", "chunk-3"},
    ) == 0.5


def test_reciprocal_rank_rejects_empty_relevant_set():
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids cannot be empty",
    ):
        reciprocal_rank(["chunk-1"], set())


def test_mrr_averages_multiple_queries():
    score = mean_reciprocal_rank(
        rankings=[
            ["a", "b"],
            ["x", "b"],
            ["x", "y"],
        ],
        relevant_chunk_ids=[
            {"a"},
            {"b"},
            {"z"},
        ],
    )

    assert score == pytest.approx(0.5)


def test_mrr_rejects_empty_rankings():
    with pytest.raises(
        ValueError,
        match="rankings cannot be empty",
    ):
        mean_reciprocal_rank([], [])


def test_mrr_rejects_mismatched_inputs():
    with pytest.raises(
        ValueError,
        match="Each ranking must have a matching",
    ):
        mean_reciprocal_rank(
            [["a"]],
            [{"a"}, {"b"}],
        )


def test_mean_hit_at_k_averages_multiple_queries():
    score = mean_hit_at_k(
        rankings=[
            ["a", "b"],
            ["x", "b"],
            ["x", "y"],
        ],
        relevant_chunk_ids=[
            {"a"},
            {"b"},
            {"z"},
        ],
        k=1,
    )

    assert score == pytest.approx(1 / 3)


def test_mean_hit_at_k_rejects_empty_rankings():
    with pytest.raises(
        ValueError,
        match="rankings cannot be empty",
    ):
        mean_hit_at_k([], [], k=1)


def test_mean_hit_at_k_rejects_mismatched_inputs():
    with pytest.raises(
        ValueError,
        match="Each ranking must have a matching",
    ):
        mean_hit_at_k(
            [["a"]],
            [{"a"}, {"b"}],
            k=1,
        )
