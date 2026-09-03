from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


import numpy as np

# text = "I love Python!!!"

# tokens = word_tokenize(text)

# # print(tokens)

# texts = [
#     "I love this product!",
#     "I don't like this product.",
#     "This is amazing 😍",
#     "This is terrible 😡",
#     "AI is changing the world."
# ]

# for text in texts:
#     print("\nOriginal:", text)
#     print("Tokens:", word_tokenize(text))

# documents = [
#     "I love Python",
#     "I love NLP",
#     "Python is powerful"
# ]

# vectorizer = CountVectorizer()

# X = vectorizer.fit_transform(documents)

# print(vectorizer.get_feature_names_out())
# print(X.toarray())    

# documents = [
#     "I love machine learning",
#     "I love deep learning",
#     "machine learning is powerful"
# ]

# vectorizer = TfidfVectorizer()

# X = vectorizer.fit_transform(documents)

# print(vectorizer.get_feature_names_out())
# print(X.toarray())

# A = np.array([[1, 2, 3]])
# B = np.array([[4, 5, 6]])

# similarity = cosine_similarity(A, B)

# print(similarity)

model = SentenceTransformer("all-MiniLM-L6-v2")

# sentence = "I love machine learning"

# embedding = model.encode(sentence)

# print(embedding)
# print(embedding.shape)

# sentences = [
#     "I love machine learning",
#     "I enjoy studying artificial intelligence",
#     "The weather is very cold today",
# ]

# embeddings = model.encode(sentences)

# print(embeddings.shape)

# similarity = cosine_similarity(embeddings)

# print(similarity)

sentences = [
    "I love this car.",
    "I really enjoy this automobile.",
    "I hate this car.",
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(embeddings)

print(similarity)