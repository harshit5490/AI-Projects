
def abstention_metrics(
    answerable: list[bool],
    abstained: list[bool],
) -> dict[str, float]:
    """
    Evaluate whether a system abstains appropriately.

    answerable[i]: whether the source documents contain the answer.
    abstained[i]: whether the system declined to answer.
    """

    if not answerable:
        raise ValueError("answerable cannot be empty.")

    if len(answerable) != len(abstained):
        raise ValueError(
            "answerable and abstained must have equal lengths."
        )

    true_abstentions = sum(
        not can_answer and did_abstain
        for can_answer, did_abstain in zip(answerable, abstained)
    )

    false_answers = sum(
        not can_answer and not did_abstain
        for can_answer, did_abstain in zip(answerable, abstained)
    )

    missed_answers = sum(
        can_answer and did_abstain
        for can_answer, did_abstain in zip(answerable, abstained)
    )

    correct_answers = sum(
        can_answer and not did_abstain
        for can_answer, did_abstain in zip(answerable, abstained)
    )

    unanswerable_count = sum(not value for value in answerable)
    answerable_count = sum(answerable)

    return {
        "abstention_accuracy": (
            (true_abstentions + correct_answers) / len(answerable)
        ),
        "false_answer_rate": (
            false_answers / unanswerable_count
            if unanswerable_count else 0.0
        ),
        "missed_answer_rate": (
            missed_answers / answerable_count
            if answerable_count else 0.0
        ),
    }
