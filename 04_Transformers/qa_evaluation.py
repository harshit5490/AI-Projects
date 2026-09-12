import re


def normalize_answer(text):
    """
    Normalize an answer before evaluation.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def exact_match(predicted, expected):
    """
    Return 1 if answers match exactly
    after normalization, otherwise 0.
    """

    predicted = normalize_answer(predicted)
    expected = normalize_answer(expected)

    return int(predicted == expected)


def f1_score(predicted, expected):
    """
    Calculate token-level F1 score.
    """

    predicted_tokens = normalize_answer(predicted).split()
    expected_tokens = normalize_answer(expected).split()

    # Handle empty answers
    if len(predicted_tokens) == 0 or len(expected_tokens) == 0:
        return int(predicted_tokens == expected_tokens)

    common_tokens = set(predicted_tokens) & set(expected_tokens)

    num_common = sum(
        min(predicted_tokens.count(token),
            expected_tokens.count(token))
        for token in common_tokens
    )

    if num_common == 0:
        return 0.0

    precision = num_common / len(predicted_tokens)
    recall = num_common / len(expected_tokens)

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    return f1


# --------------------------------------------------
# Test examples
# --------------------------------------------------

examples = [
    ("Guido van Rossum", "Guido van Rossum"),
    ("guido van rossum", "Guido van Rossum"),
    ("Guido van", "Guido van Rossum"),
    ("Rossum", "Guido van Rossum"),
    ("Java", "Guido van Rossum"),
]


for predicted, expected in examples:

    em = exact_match(predicted, expected)
    f1 = f1_score(predicted, expected)

    print("\nPredicted:", predicted)
    print("Expected :", expected)
    print("EM       :", em)
    print("F1       :", round(f1, 4))