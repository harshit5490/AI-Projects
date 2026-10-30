from pathlib import Path
from uuid import uuid4

from app.chunking.recursive import RecursiveCharacterChunker
from app.ingestion.factory import DocumentParserFactory
from app.services.chunk_service import ChunkService
from app.services.ingestion_service import IngestionService


class DocumentService:

    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def __init__(
        self,
        ingestion_service: IngestionService,
        chunker: RecursiveCharacterChunker,
    ):
        self.ingestion_service = ingestion_service
        self.chunk_service = ChunkService(chunker=chunker)

    def process_document(
        self,
        file_path: str,
        filename: str,
    ) -> dict:

        extension = Path(filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        document_id = str(uuid4())

        parser = DocumentParserFactory.get_parser(file_path)

        parsed_documents = parser.parse(file_path)

        if not parsed_documents:
            raise ValueError(
                "The document does not contain readable text."
            )

        file_type = extension.lstrip(".")

        chunks = self.chunk_service.create_chunks(
            parsed_documents=parsed_documents,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
        )

        chunk_count = self.ingestion_service.ingest_chunks(chunks)

        return {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "chunk_count": chunk_count,
        }