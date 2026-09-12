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
# 2. Question and context
# --------------------------------------------------

question = "Who created Python?"

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


# --------------------------------------------------
# 3. Tokenization
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
# 4. Identify question/context tokens
# --------------------------------------------------

sequence_ids = inputs.sequence_ids()


print("=" * 60)
print("TOKEN SEQUENCE")
print("=" * 60)

for i, token in enumerate(tokens):

    print(
        i,
        "->",
        token,
        "-> sequence:",
        sequence_ids[i]
    )


# --------------------------------------------------
# 5. Model inference
# --------------------------------------------------

model.eval()

with torch.no_grad():

    outputs = model(**inputs)


start_logits = outputs.start_logits[0]

end_logits = outputs.end_logits[0]


# --------------------------------------------------
# 6. Find context boundaries
# --------------------------------------------------

context_start = None
context_end = None


for i, sequence_id in enumerate(sequence_ids):

    if sequence_id == 1:

        if context_start is None:
            context_start = i

        context_end = i


print("\nContext Start:")
print(context_start)

print("\nContext End:")
print(context_end)


# --------------------------------------------------
# 7. Find best valid context span
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
# 8. Extract answer
# --------------------------------------------------

answer_tokens = tokens[
    best_start:best_end + 1
]


answer = tokenizer.convert_tokens_to_string(
    answer_tokens
)


# --------------------------------------------------
# 9. Display result
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("\nBest Start Position:")
print(best_start)

print("\nBest End Position:")
print(best_end)

print("\nBest Span Score:")
print(round(best_score, 4))

print("\nAnswer:")
print(answer)