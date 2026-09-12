import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


MODEL_NAME = "distilbert-base-uncased-distilled-squad"


# --------------------------------------------------
# 1. Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# 2. Load Question Answering model
# --------------------------------------------------

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# 3. Context
# --------------------------------------------------

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


# --------------------------------------------------
# 4. Questions
# --------------------------------------------------

questions = [
    "Who created Python?",
    "When was Python first released?",
    "What is Python used for?",
    "Who invented Java?"
]


print("=" * 60)
print("QUESTION ANSWERING SYSTEM")
print("=" * 60)


# --------------------------------------------------
# 5. Process every question
# --------------------------------------------------

model.eval()

for question in questions:

    print("\nQuestion:")
    print(question)

    # Tokenize question + context
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt"
    )

    # Model inference
    with torch.no_grad():

        outputs = model(**inputs)

    # Find most likely start and end positions
    start_position = torch.argmax(
        outputs.start_logits,
        dim=-1
    ).item()

    end_position = torch.argmax(
        outputs.end_logits,
        dim=-1
    ).item()

    # Convert token IDs back to text
    answer_tokens = inputs["input_ids"][
        0,
        start_position:end_position + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    print("\nAnswer:")
    print(answer)

    # Calculate confidence-like values
    start_score = torch.softmax(
        outputs.start_logits,
        dim=-1
    )[0, start_position].item()

    end_score = torch.softmax(
        outputs.end_logits,
        dim=-1
    )[0, end_position].item()

    print("\nStart Score:")
    print(round(start_score, 4))

    print("End Score:")
    print(round(end_score, 4))

    print("-" * 60)