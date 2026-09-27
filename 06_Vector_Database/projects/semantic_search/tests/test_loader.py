from app.ingestion.loader import load_documents

documents = load_documents(
    "data/documents"
)

for document in documents:

    print("\nID:", document.id)
    print("Source:", document.source)
    print("Category:", document.category)
    print("Text:", document.text)