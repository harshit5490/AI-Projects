import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)


MODEL_NAME = "distilbert-base-uncased-distilled-squad"


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
    return_tensors="pt",
    return_offsets_mapping=True
)


tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

offsets = inputs["offset_mapping"][0]

sequence_ids = inputs.sequence_ids()


# --------------------------------------------------
# Model inference
# --------------------------------------------------

model.eval()

with torch.no_grad():

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )


# --------------------------------------------------
# Start and end predictions
# --------------------------------------------------

start_position = torch.argmax(
    outputs.start_logits,
    dim=-1
).item()

end_position = torch.argmax(
    outputs.end_logits,
    dim=-1
).item()


# --------------------------------------------------
# Convert token positions → character positions
# --------------------------------------------------

start_char = offsets[start_position][0].item()

end_char = offsets[end_position][1].item()


# --------------------------------------------------
# Extract answer directly from original context
# --------------------------------------------------

answer = context[
    start_char:end_char
]


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("=" * 60)
print("OFFSET-BASED QA")
print("=" * 60)

print("\nQuestion:")
print(question)

print("\nStart Token:")
print(start_position, "->", tokens[start_position])

print("\nEnd Token:")
print(end_position, "->", tokens[end_position])

print("\nCharacter Start:")
print(start_char)

print("\nCharacter End:")
print(end_char)

print("\nAnswer:")
print(answer)