import pymupdf

from app.ingestion.base import BaseDocumentParser


class PDFParser(BaseDocumentParser):

    def parse(self, file_path: str) -> list:
        documents = []

        pdf = pymupdf.open(file_path)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            if text.strip():
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": file_path,
                            "page": page_number,
                            "file_type": "pdf",
                        },
                    }
                )

        pdf.close()

        return documents