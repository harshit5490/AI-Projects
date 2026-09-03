import numpy as np
import pandas as pd

from pathlib import Path
from src.preprocessing import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from src.embedding_model import (
    load_embedding_model,
    generate_embeddings
)

BASE_DIR = Path(__file__).parent.parent

df = pd.read_csv(BASE_DIR/"data"/"raw"/"IMDB Dataset.csv")

# print(df.shape)

# print(df.columns)

# print(df.head())

# df.info()

# print(df.isnull().sum())

# print("Duplicates:", df.duplicated().sum())

# print(df["sentiment"].value_counts())

# print(df["sentiment"].value_counts(normalize=True) * 100)

# print(df["sentiment"].unique())

# df["text_length"] = df["review"].astype(str).str.len()

# df["word_count"] = (
#     df["review"]
#     .astype(str)
#     .str.split()
#     .str.len()
# )

# print(df["text_length"].describe())
# print(df["word_count"].describe())

# duplicates = df[df.duplicated(keep=False)]

# print(duplicates.shape)

# duplicates.head(10)

# duplicate_reviews = (
#     df.groupby("review")["sentiment"]
#     .nunique()
# )

# print(
#     duplicate_reviews.value_counts()
# )

# # Create word count
# df["word_count"] = df["review"].str.split().str.len()

# # 1. Duplicate label consistency
# duplicate_reviews = df.groupby("review")["sentiment"].nunique()
# print(duplicate_reviews.value_counts())


# # 2. HTML
# html_count = df["review"].str.contains(
#     r"<[^>]+>", regex=True, na=False
# ).sum()

# print("HTML:", html_count)
# print("HTML %:", html_count / len(df) * 100)


# # 3. URLs
# url_count = df["review"].str.contains(
#     r"https?://|www\.",
#     regex=True,
#     case=False,
#     na=False
# ).sum()

# print("URLs:", url_count)
# print("URLs %:", url_count / len(df) * 100)


# # 4. Longest reviews
# print(df.nlargest(5, "word_count")[["sentiment", "word_count", "review"]])

# negation_words = [
#     "not",
#     "no",
#     "never",
#     "don't",
#     "didn't",
#     "isn't",
#     "wasn't",
#     "couldn't",
#     "wouldn't",
#     "won't"
# ]

# for word in negation_words:
#     count = df["review"].str.contains(
#         rf"\b{word}\b",
#         case=False,
#         regex=True,
#         na=False
#     ).sum()

#     print(f"{word:10} : {count}")

# exclamation_count = df["review"].str.contains(
#     "!",
#     regex=False
# ).sum()

# question_count = df["review"].str.contains(
#     "?",
#     regex=False
# ).sum()

# print("Reviews containing !:", exclamation_count)
# print("Reviews containing ?:", question_count)    

df["clean_review"] = df["review"].apply(clean_text)

df = df.drop_duplicates(
    subset="clean_review"
).reset_index(drop=True)

print("Shape:", df.shape)

print("\nClass distribution:")
print(df["sentiment"].value_counts())

print("\nMissing cleaned reviews:")
print(df["clean_review"].isnull().sum())

print("\nEmpty cleaned reviews:")
print((df["clean_review"].str.len() == 0).sum())

print("\nExamples:")
print(df[["review", "clean_review"]].head())

X = df["clean_review"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting distribution:")
print(y_test.value_counts(normalize=True))

# vectorizer = TfidfVectorizer()

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# print("Training TF-IDF shape:", X_train_tfidf.shape)
# print("Testing TF-IDF shape:", X_test_tfidf.shape)

# print("Vocabulary size:", len(vectorizer.vocabulary_))

# TF-IDF
# vectorizer = TfidfVectorizer()

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# print("Training TF-IDF shape:", X_train_tfidf.shape)
# print("Testing TF-IDF shape:", X_test_tfidf.shape)
# print("Vocabulary size:", len(vectorizer.vocabulary_))


# Logistic Regression
# model = LogisticRegression(
#     max_iter=1000
# )

# model.fit(
#     X_train_tfidf,
#     y_train
# )


# # Prediction
# y_pred = model.predict(
#     X_test_tfidf
# )


# # Accuracy
# accuracy = accuracy_score(
#     y_test,
#     y_pred
# )

# print("\nAccuracy:", accuracy)


# # Classification report
# print("\nClassification Report:")
# print(
#     classification_report(
#         y_test,
#         y_pred
#     )
# )


# # Confusion matrix
# print("\nConfusion Matrix:")
# print(
#     confusion_matrix(
#         y_test,
#         y_pred
#     )
# )

# results = pd.DataFrame({
#     "review": X_test,
#     "actual": y_test,
#     "predicted": y_pred
# })

# errors = results[
#     results["actual"] != results["predicted"]
# ]

# print("Total errors:", len(errors))

# print("\nError distribution:")
# print(
#     errors.groupby(
#         ["actual", "predicted"]
#     ).size()
# )

# print("\nSample errors:")
# print(
#     errors.sample(
#         10,
#         random_state=42
#     ).to_string(index=False)
# )



embedding_model = load_embedding_model()

# sample_texts = X_train.iloc[:5].tolist()

# sample_embeddings = generate_embeddings(
#     embedding_model,
#     sample_texts
# )

# print("Embedding shape:", sample_embeddings.shape)




# sentences = [
#     "I absolutely loved this movie.",
#     "This film was fantastic.",
#     "I hated this movie.",
#     "This film was terrible."
# ]

# embeddings = generate_embeddings(
#     embedding_model,
#     sentences
# )

# similarity = cosine_similarity(embeddings)

# print("\nSimilarity Matrix:")
# print(similarity)

X_train_embeddings = generate_embeddings(
    embedding_model,
    X_train.tolist()
)

X_test_embeddings = generate_embeddings(
    embedding_model,
    X_test.tolist()
)

print("Train embedding shape:", X_train_embeddings.shape)
print("Test embedding shape:", X_test_embeddings.shape)

embedding_model_lr = LogisticRegression(
    max_iter=1000
)

embedding_model_lr.fit(
    X_train_embeddings,
    y_train
)

y_pred_embeddings = embedding_model_lr.predict(
    X_test_embeddings
)
accuracy_embeddings = accuracy_score(
    y_test,
    y_pred_embeddings
)

print("Embedding Accuracy:", accuracy_embeddings)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_embeddings
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_embeddings
    )
)