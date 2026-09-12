import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# --------------------------------------------------
# 1. Load model and tokenizer
# --------------------------------------------------

MODEL_NAME = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

model.eval()


# --------------------------------------------------
# 2. Context
# --------------------------------------------------

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


# --------------------------------------------------
# 3. Questions
# --------------------------------------------------

questions = [
    "Who created Python?",
    "When was Python first released?",
    "What is Python used for?",
    "Who invented Java?"
]


# --------------------------------------------------
# 4. Configuration
# --------------------------------------------------

MAX_ANSWER_LENGTH = 15

# Educational baseline.
# In production, this should be tuned using
# a validation dataset.
NULL_SCORE_DIFF_THRESHOLD = 0.0


# --------------------------------------------------
# 5. Process each question
# --------------------------------------------------

for question in questions:

    print("\n" + "=" * 60)
    print("Question:", question)

    # Tokenize question + context
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    # Get sequence IDs
    # 0 = question
    # 1 = context
    # None = special tokens
    sequence_ids = inputs.sequence_ids()

    # --------------------------------------------------
    # 6. Find context token boundaries
    # --------------------------------------------------

    context_positions = [
        i for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    context_start = context_positions[0]
    context_end = context_positions[-1]

    # --------------------------------------------------
    # 7. Model inference
    # --------------------------------------------------

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # --------------------------------------------------
    # 8. Null answer score
    # --------------------------------------------------

    null_score = (
        start_logits[0].item()
        + end_logits[0].item()
    )

    # --------------------------------------------------
    # 9. Find best valid non-null span
    # --------------------------------------------------

    best_start = None
    best_end = None
    best_score = float("-inf")

    for start in range(context_start, context_end + 1):

        for end in range(start, context_end + 1):

            # Limit answer length
            if end - start + 1 > MAX_ANSWER_LENGTH:
                break

            score = (
                start_logits[start].item()
                + end_logits[end].item()
            )

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    # --------------------------------------------------
    # 10. Compare null answer with best answer
    # --------------------------------------------------

    score_difference = null_score - best_score

    # --------------------------------------------------
    # 11. Decide whether answer exists
    # --------------------------------------------------

    if score_difference >= NULL_SCORE_DIFF_THRESHOLD:

        answer = "No answer found in the provided context."

        print("Answer:", answer)
        print("Decision: NO ANSWER")

    else:

        # --------------------------------------------------
        # 12. Convert token positions to character positions
        # --------------------------------------------------

        offsets = inputs["offset_mapping"][0]

        start_char = offsets[best_start][0].item()
        end_char = offsets[best_end][1].item()

        # Extract exact text from original context
        answer = context[start_char:end_char]

        print("Answer:", answer)
        print("Decision: ANSWER FOUND")

    # --------------------------------------------------
    # 13. Debug information
    # --------------------------------------------------

    print("Context Start Position:", context_start)
    print("Context End Position:", context_end)

    print("Best Start Position:", best_start)
    print("Best End Position:", best_end)

    print("Best Span Score:", round(best_score, 4))
    print("Null Score:", round(null_score, 4))
    print("Score Difference:", round(score_difference, 4))