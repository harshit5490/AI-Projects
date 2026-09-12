import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


MODEL_NAME = "deepset/roberta-base-squad2"


# --------------------------------------------------
# 1. Load tokenizer and model
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

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
        return_tensors="pt",
        return_offsets_mapping=True
    )


    # --------------------------------------------------
    # Model inference
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )


    start_logits = outputs.start_logits[0]

    end_logits = outputs.end_logits[0]


    # --------------------------------------------------
    # Find best start and end positions
    # --------------------------------------------------

    start_position = torch.argmax(
        start_logits
    ).item()

    end_position = torch.argmax(
        end_logits
    ).item()


    # --------------------------------------------------
    # Calculate null score
    # --------------------------------------------------

    null_score = (
        start_logits[0].item()
        + end_logits[0].item()
    )


    # --------------------------------------------------
    # Calculate best span score
    # --------------------------------------------------

    best_score = float("-inf")

    best_start = None
    best_end = None


    for start in range(
        len(inputs["input_ids"][0])
    ):

        for end in range(
            start,
            len(inputs["input_ids"][0])
        ):

            answer_length = end - start + 1


            if answer_length > 15:
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
    # Display raw scores
    # --------------------------------------------------

    print("\nBest Span Score:")
    print(round(best_score, 4))

    print("\nNull Score:")
    print(round(null_score, 4))

    print("\nStart Position:")
    print(start_position)

    print("\nEnd Position:")
    print(end_position)