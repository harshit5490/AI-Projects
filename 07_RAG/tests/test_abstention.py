
import pytest

from app.evaluation.abstention_metrics import abstention_metrics


def test_perfect_abstention_behavior():
    result = abstention_metrics(
        answerable=[True, True, False, False],
        abstained=[False, False, True, True],
    )

    assert result["abstention_accuracy"] == 1.0
    assert result["false_answer_rate"] == 0.0
    assert result["missed_answer_rate"] == 0.0


def test_system_answers_everything():
    result = abstention_metrics(
        answerable=[True, False],
        abstained=[False, False],
    )

    assert result["abstention_accuracy"] == 0.5
    assert result["false_answer_rate"] == 1.0
    assert result["missed_answer_rate"] == 0.0


def test_system_abstains_everywhere():
    result = abstention_metrics(
        answerable=[True, False],
        abstained=[True, True],
    )

    assert result["abstention_accuracy"] == 0.5
    assert result["false_answer_rate"] == 0.0
    assert result["missed_answer_rate"] == 1.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        abstention_metrics([True, False], [True])


def test_empty_input_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        abstention_metrics([], [])
