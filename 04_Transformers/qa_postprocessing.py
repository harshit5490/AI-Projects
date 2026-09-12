import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


MODEL_NAME = "distilbert-base-uncased-distilled-squad"

MAX_ANSWER_LENGTH = 15


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)


question = "Who created Python?"

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


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


# --------------------------------------------------
# Model inference
# --------------------------------------------------

model.eval()

with torch.no_grad():

    outputs = model(**inputs)


start_logits = outputs.start_logits[0]

end_logits = outputs.end_logits[0]


# --------------------------------------------------
# Find best valid answer span
# --------------------------------------------------

best_score = float("-inf")

best_start = None
best_end = None


for start in range(len(tokens)):

    for end in range(start, len(tokens)):

        # Limit answer length
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
# Extract answer
# --------------------------------------------------

answer_tokens = tokens[
    best_start:best_end + 1
]


answer = tokenizer.convert_tokens_to_string(
    answer_tokens
)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("=" * 60)
print("QA POST-PROCESSING")
print("=" * 60)

print("\nQuestion:")
print(question)

print("\nBest Start Position:")
print(best_start)

print("\nBest End Position:")
print(best_end)

print("\nBest Span Score:")
print(round(best_score, 4))

print("\nAnswer:")
print(answer)