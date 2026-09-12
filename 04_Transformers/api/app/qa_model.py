import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

# MODEL_NAME = "deepset/roberta-base-squad2"
# MODEL_NAME = "deepset/tinyroberta-squad2"
MODEL_NAME = "deepset/minilm-uncased-squad2"

MAX_ANSWER_LENGTH = 15

# Educational baseline.
# In production, tune this using validation data.
NULL_SCORE_DIFF_THRESHOLD = 0.0


# ============================================================
# 2. LOAD TOKENIZER AND MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

model.eval()


# ============================================================
# 3. QA PREDICTION FUNCTION
# ============================================================

def predict_answer(question, context):

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
    # Identify question/context tokens
    # --------------------------------------------------------

    sequence_ids = inputs.sequence_ids()

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
    # Get start and end logits
    # --------------------------------------------------------

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # --------------------------------------------------------
    # Calculate null answer score
    # --------------------------------------------------------

    null_score = (
        start_logits[0].item()
        + end_logits[0].item()
    )

    # --------------------------------------------------------
    # Find best valid answer span
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

    # --------------------------------------------------------
    # Compare null answer with best answer
    # --------------------------------------------------------

    score_difference = null_score - best_score

    # --------------------------------------------------------
    # No-answer decision
    # --------------------------------------------------------

    if score_difference >= NULL_SCORE_DIFF_THRESHOLD:

        return {
            "answer": "",
            "has_answer": False,
            "best_span_score": round(best_score, 4),
            "null_score": round(null_score, 4),
            "score_difference": round(score_difference, 4)
        }

    # --------------------------------------------------------
    # Extract exact answer using offsets
    # --------------------------------------------------------

    offsets = inputs["offset_mapping"][0]

    start_char = offsets[best_start][0].item()
    end_char = offsets[best_end][1].item()

    answer = context[start_char:end_char]

    return {
        "answer": answer,
        "has_answer": True,
        "best_span_score": round(best_score, 4),
        "null_score": round(null_score, 4),
        "score_difference": round(score_difference, 4)
    }