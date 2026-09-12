import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


MODEL_NAME = "distilbert-base-uncased-distilled-squad"


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


print("=" * 70)
print("QA MODEL INSPECTION")
print("=" * 70)


# --------------------------------------------------
# 4. Show input IDs
# --------------------------------------------------

print("\nInput IDs:")
print(inputs["input_ids"])


# --------------------------------------------------
# 5. Convert IDs to tokens
# --------------------------------------------------

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)


print("\nTokens:")

for index, token in enumerate(tokens):
    print(index, "->", token)


# --------------------------------------------------
# 6. Model inference
# --------------------------------------------------

model.eval()

with torch.no_grad():

    outputs = model(**inputs)


# --------------------------------------------------
# 7. Start logits
# --------------------------------------------------

print("\nStart Logits:")
print(outputs.start_logits)


# --------------------------------------------------
# 8. End logits
# --------------------------------------------------

print("\nEnd Logits:")
print(outputs.end_logits)


# --------------------------------------------------
# 9. Find start and end positions
# --------------------------------------------------

start_position = torch.argmax(
    outputs.start_logits,
    dim=-1
).item()


end_position = torch.argmax(
    outputs.end_logits,
    dim=-1
).item()


print("\nPredicted Start Position:")
print(start_position)


print("\nPredicted End Position:")
print(end_position)


# --------------------------------------------------
# 10. Extract answer tokens
# --------------------------------------------------

answer_tokens = tokens[
    start_position:end_position + 1
]


print("\nAnswer Tokens:")
print(answer_tokens)


# --------------------------------------------------
# 11. Convert tokens back to text
# --------------------------------------------------

answer = tokenizer.convert_tokens_to_string(
    answer_tokens
)


print("\nFinal Answer:")
print(answer)