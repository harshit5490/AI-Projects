from docx import Document

from app.ingestion.base import BaseDocumentParser


class DOCXParser(BaseDocumentParser):

    def parse(self, file_path: str) -> list:
        document = Document(file_path)

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        if not paragraphs:
            return []

        text = "\n\n".join(paragraphs)

        return [
            {
                "text": text,
                "metadata": {
                    "source": file_path,
                    "file_type": "docx",
                },
            }
        ]