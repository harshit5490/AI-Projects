
from pathlib import Path

from app.chunking.recursive import RecursiveCharacterChunker
from app.embeddings.service import EmbeddingService
from app.ingestion.txt_parser import TXTParser
from app.services.chunk_service import ChunkService
from app.services.ingestion_service import IngestionService
from app.vectorstore.store import PineconeVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_documents.txt"
)

EVALUATION_NAMESPACE = "rag-evaluation"
EVALUATION_DOCUMENT_ID = "eval-knowledge-v1"
FILENAME = "evaluation_documents.txt"


def main() -> None:
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation document not found: {DOCUMENT_PATH}"
        )

    embedding_service = EmbeddingService()

    vector_store = PineconeVectorStore(
        index_name="rag-document-index",
        namespace=EVALUATION_NAMESPACE,
    )

    chunker = RecursiveCharacterChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunk_service = ChunkService(chunker=chunker)
    parser = TXTParser()

    parsed_documents = parser.parse(str(DOCUMENT_PATH))

    chunks = chunk_service.create_chunks(
        parsed_documents=parsed_documents,
        document_id=EVALUATION_DOCUMENT_ID,
        filename=FILENAME,
        file_type="txt",
    )

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    count = ingestion_service.ingest_chunks(chunks)

    print(f"Evaluation chunks ingested: {count}")
    print(f"Namespace: {EVALUATION_NAMESPACE}")

    for chunk in chunks:
        chunk_id = (
            f"{chunk.document_id}"
            f"_chunk_{chunk.chunk_index:04d}"
        )
        print(f"\nChunk ID: {chunk_id}")
        print(chunk.text)


if __name__ == "__main__":
    main()
