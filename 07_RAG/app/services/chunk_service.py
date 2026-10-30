from app.schemas.chunk import DocumentChunk


class ChunkService:

    def __init__(
        self,
        chunker,
    ):
        self.chunker = chunker

    def create_chunks(
        self,
        parsed_documents: list[dict],
        document_id: str,
        filename: str,
        file_type: str,
    ) -> list[DocumentChunk]:

        chunks = []
        chunk_index = 0

        for parsed_document in parsed_documents:

            text = parsed_document["text"]

            source_metadata = parsed_document.get(
                "metadata",
                {},
            )

            page = source_metadata.get("page")

            text_chunks = self.chunker.split(text)

            for chunk_text in text_chunks:

                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        file_type=file_type,
                        chunk_index=chunk_index,
                        page=page,
                    )
                )

                chunk_index += 1

        return chunks