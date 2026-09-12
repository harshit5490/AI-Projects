import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


MODEL_NAME = "distilbert-base-uncased-distilled-squad"

MAX_ANSWER_LENGTH = 15


# --------------------------------------------------
# 1. Load tokenizer and model
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# 2. Context
# --------------------------------------------------

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


questions = [
    "Who created Python?",
    "When was Python first released?",
    "What is Python used for?",
    "Who invented Java?"
]


# --------------------------------------------------
# 3. Process questions
# --------------------------------------------------

model.eval()


for question in questions:

    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt"
    )

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    sequence_ids = inputs.sequence_ids()


    # --------------------------------------------------
    # Model inference
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)


    start_logits = outputs.start_logits[0]

    end_logits = outputs.end_logits[0]


    # --------------------------------------------------
    # Find context boundaries
    # --------------------------------------------------

    context_start = None
    context_end = None


    for i, sequence_id in enumerate(sequence_ids):

        if sequence_id == 1:

            if context_start is None:
                context_start = i

            context_end = i


    # --------------------------------------------------
    # Find best answer span
    # --------------------------------------------------

    best_score = float("-inf")

    best_start = None
    best_end = None


    for start in range(
        context_start,
        context_end + 1
    ):

        for end in range(
            start,
            context_end + 1
        ):

            answer_length = end - start + 1

            if answer_length > MAX_ANSWER_LENGTH:
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
    # Calculate null score
    # --------------------------------------------------

    null_score = (
        start_logits[0].item()
        + end_logits[0].item()
    )


    # --------------------------------------------------
    # Compare answer vs no-answer
    # --------------------------------------------------

    print("\nBest Answer Span Score:")
    print(round(best_score, 4))

    print("\nNull Score:")
    print(round(null_score, 4))


    if best_score > null_score:

        answer_tokens = tokens[
            best_start:best_end + 1
        ]

        answer = tokenizer.convert_tokens_to_string(
            answer_tokens
        )

        print("\nAnswer:")
        print(answer)

    else:

        print("\nAnswer:")
        print("No answer found in the provided context.")