import re
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ============================================================
# 1. LOAD MODEL AND TOKENIZER
# ============================================================

MODEL_NAME = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

model.eval()


# ============================================================
# 2. CONTEXT
# ============================================================

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


# ============================================================
# 3. QUESTIONS AND EXPECTED ANSWERS
# ============================================================

examples = [
    {
        "question": "Who created Python?",
        "expected": "Guido van Rossum"
    },
    {
        "question": "When was Python first released?",
        "expected": "1991"
    },
    {
        "question": "What is Python used for?",
        "expected": (
            "web development, data science, automation, "
            "machine learning, and artificial intelligence"
        )
    },
    {
        "question": "Who invented Java?",
        "expected": ""
    }
]


# ============================================================
# 4. CONFIGURATION
# ============================================================

MAX_ANSWER_LENGTH = 15

# Educational baseline.
# In production, this threshold should be tuned
# using a validation dataset.

NULL_SCORE_DIFF_THRESHOLD = 0.0


# ============================================================
# 5. NORMALIZE ANSWER
# ============================================================

def normalize_answer(text):
    """
    Normalize answer text before evaluation.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


# ============================================================
# 6. EXACT MATCH
# ============================================================

def exact_match(predicted, expected):
    """
    Returns 1 if predicted and expected answers
    are exactly equal after normalization.
    Otherwise returns 0.
    """

    predicted = normalize_answer(predicted)
    expected = normalize_answer(expected)

    return int(predicted == expected)


# ============================================================
# 7. F1 SCORE
# ============================================================

def f1_score(predicted, expected):
    """
    Calculate token-level F1 score.
    """

    predicted_tokens = normalize_answer(predicted).split()
    expected_tokens = normalize_answer(expected).split()

    # Handle empty answers
    if len(predicted_tokens) == 0 or len(expected_tokens) == 0:
        return int(predicted_tokens == expected_tokens)

    # Find common tokens
    common_tokens = set(predicted_tokens) & set(expected_tokens)

    num_common = sum(
        min(
            predicted_tokens.count(token),
            expected_tokens.count(token)
        )
        for token in common_tokens
    )

    # No overlap
    if num_common == 0:
        return 0.0

    # Precision
    precision = num_common / len(predicted_tokens)

    # Recall
    recall = num_common / len(expected_tokens)

    # F1
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    return f1


# ============================================================
# 8. QA PREDICTION FUNCTION
# ============================================================

def predict_answer(question, context):
    """
    Predict an answer from the provided context.

    Returns:
        answer
        best span score
        null score
        score difference
    """

    # --------------------------------------------------------
    # Tokenize question + context
    # --------------------------------------------------------

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    # --------------------------------------------------------
    # Get sequence IDs
    #
    # 0    -> question
    # 1    -> context
    # None -> special tokens
    # --------------------------------------------------------

    sequence_ids = inputs.sequence_ids()

    # --------------------------------------------------------
    # Find context token positions
    # --------------------------------------------------------

    context_positions = [
        i
        for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    context_start = context_positions[0]
    context_end = context_positions[-1]

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # --------------------------------------------------------
    # Extract logits
    # --------------------------------------------------------

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # --------------------------------------------------------
    # Calculate null answer score
    #
    # Position 0 represents the special starting token.
    # --------------------------------------------------------

    null_score = (
        start_logits[0].item()
        + end_logits[0].item()
    )

    # --------------------------------------------------------
    # Find best valid context span
    # --------------------------------------------------------

    best_start = None
    best_end = None

    best_score = float("-inf")

    for start in range(
        context_start,
        context_end + 1
    ):

        for end in range(
            start,
            context_end + 1
        ):

            # Limit maximum answer length
            if end - start + 1 > MAX_ANSWER_LENGTH:
                break

            # Calculate span score
            score = (
                start_logits[start].item()
                + end_logits[end].item()
            )

            # Keep best span
            if score > best_score:

                best_score = score
                best_start = start
                best_end = end

    # --------------------------------------------------------
    # Compare null answer vs best answer
    # --------------------------------------------------------

    score_difference = null_score - best_score

    # --------------------------------------------------------
    # Decide answer / no-answer
    # --------------------------------------------------------

    if score_difference >= NULL_SCORE_DIFF_THRESHOLD:

        answer = ""

    else:

        # ----------------------------------------------------
        # Convert token positions to character positions
        # ----------------------------------------------------

        offsets = inputs["offset_mapping"][0]

        start_char = offsets[best_start][0].item()
        end_char = offsets[best_end][1].item()

        # ----------------------------------------------------
        # Extract exact answer from original context
        # ----------------------------------------------------

        answer = context[start_char:end_char]

    return (
        answer,
        best_score,
        null_score,
        score_difference
    )


# ============================================================
# 9. EVALUATE ALL QUESTIONS
# ============================================================

total_em = 0
total_f1 = 0.0


for example in examples:

    question = example["question"]
    expected = example["expected"]

    # --------------------------------------------------------
    # Get model prediction
    # --------------------------------------------------------

    (
        predicted,
        best_score,
        null_score,
        score_difference
    ) = predict_answer(
        question,
        context
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    em = exact_match(
        predicted,
        expected
    )

    f1 = f1_score(
        predicted,
        expected
    )

    # Add to totals
    total_em += em
    total_f1 += f1

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("Question :", question)

    print("Expected :", expected if expected else "NO ANSWER")

    print(
        "Predicted:",
        predicted if predicted else "NO ANSWER"
    )

    print("EM       :", em)
    print("F1       :", round(f1, 4))

    print("Best Span Score :", round(best_score, 4))
    print("Null Score      :", round(null_score, 4))
    print("Score Difference:", round(score_difference, 4))


# ============================================================
# 10. OVERALL EVALUATION
# ============================================================

number_of_examples = len(examples)

average_em = total_em / number_of_examples
average_f1 = total_f1 / number_of_examples


print("\n" + "=" * 70)
print("FINAL QA EVALUATION")
print("=" * 70)

print("Number of Examples:", number_of_examples)

print("Average EM:", round(average_em, 4))
print("Average F1:", round(average_f1, 4))