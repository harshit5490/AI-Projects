from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-distilled-squad"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


question = "Who created Python?"

context = """
Python was created by Guido van Rossum and first released
in 1991. Python is widely used in web development, data
science, automation, machine learning, and artificial intelligence.
"""


# --------------------------------------------------
# Tokenize with offset mapping
# --------------------------------------------------

inputs = tokenizer(
    question,
    context,
    return_offsets_mapping=True
)


tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"]
)


offsets = inputs["offset_mapping"]


sequence_ids = inputs.sequence_ids()


# --------------------------------------------------
# Display tokens and offsets
# --------------------------------------------------

print("=" * 70)
print("TOKEN OFFSETS")
print("=" * 70)


for i, (token, offset, sequence_id) in enumerate(
    zip(tokens, offsets, sequence_ids)
):

    print(
        f"{i:2} -> "
        f"{token:15} "
        f"offset={tuple(offset)} "
        f"sequence={sequence_id}"
    )