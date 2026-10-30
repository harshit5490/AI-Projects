
import pytest

from app.evaluation.abstention_policy import should_abstain


def test_abstains_when_no_chunks_are_retrieved():
    assert should_abstain([]) is True


def test_does_not_abstain_when_chunk_meets_threshold():
    chunks = [{"score": 0.65}]
    assert should_abstain(chunks, 0.30) is False


def test_abstains_when_all_chunks_are_below_threshold():
    chunks = [{"score": 0.20}, {"score": 0.25}]
    assert should_abstain(chunks, 0.30) is True


def test_threshold_boundary_is_inclusive():
    assert should_abstain(
        [{"score": 0.30}],
        0.30,
    ) is False


def test_rejects_invalid_threshold():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        should_abstain([{"score": 0.50}], 1.5)
