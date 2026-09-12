import torch
from transformers import AutoTokenizer, AutoModel

# model_name = "bert-base-uncased"

# # -------------------------
# # 1. Select device
# # -------------------------

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print("Using device:", device)

# # -------------------------
# # 2. Load tokenizer
# # -------------------------

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # -------------------------
# # 3. Load pretrained model
# # -------------------------

# model = AutoModel.from_pretrained(model_name)

# # -------------------------
# # 4. Move model to device
# # -------------------------

# model = model.to(device)

# # -------------------------
# # 5. Evaluation mode
# # -------------------------

# model.eval()

# # -------------------------
# # 6. Input text
# # -------------------------

# text = "I love machine learning"

# # -------------------------
# # 7. Tokenize
# # -------------------------

# inputs = tokenizer(
#     text,
#     return_tensors="pt"
# )

# # -------------------------
# # 8. Move inputs to device
# # -------------------------

# inputs = {
#     key: value.to(device)
#     for key, value in inputs.items()
# }

# # -------------------------
# # 9. Inference
# # -------------------------

# with torch.no_grad():
#     outputs = model(**inputs)

# # -------------------------
# # 10. Inspect output
# # -------------------------

# print(
#     "Hidden state shape:",
#     outputs.last_hidden_state.shape
# )

from transformers import pipeline


print("CUDA available:", torch.cuda.is_available())


# --------------------------------
# 1. Sentiment Analysis
# --------------------------------

# classifier = pipeline(
#     "sentiment-analysis",
#     device=-1
# )

# texts = [
#     "I love machine learning.",
#     "This is a terrible experience.",
#     "Transformers are interesting."
# ]

# results = classifier(texts)

# print("\nSentiment Results:")

# for text, result in zip(texts, results):
#     print(f"{text}")
#     print(result)
#     print()


# # --------------------------------
# # 2. Fill Mask
# # --------------------------------

# fill_mask = pipeline(
#     "fill-mask",
#     model="bert-base-uncased",
#     device=-1
# )

# result = fill_mask(
#     "The capital of France is [MASK]."
# )

# print("\nFill Mask Result:")
# print(result[:3])


# --------------------------------
# 3. Question Answering
# --------------------------------

# qa = pipeline(
#     "question-answering",
#     device=-1
# )

context = """
Paris is the capital and most populous city of France.
It is located along the River Seine.
"""

question = "What is the capital of France?"
generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    device=-1
)

prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

result = generator(
    prompt,
    max_new_tokens=20
)

print("\nQA Result:")
print(result)