from pathlib import Path

from app.models import Document


CATEGORY_MAP = {
    "python": "programming",
    "fastapi": "backend",
    "machine_learning": "AI",
    "transformers": "AI",
    "docker": "devops",
}


def load_documents(directory: str) -> list[Document]:
    """
    Load all .txt files from a directory.

    Each text file becomes one Document object.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    documents = []

    for file_path in sorted(directory_path.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        document_id = file_path.stem

        category = CATEGORY_MAP.get(
            document_id,
            "other",
        )

        document = Document(
            id=document_id,
            text=text,
            source=file_path.name,
            category=category,
        )

        documents.append(document)

    return documents