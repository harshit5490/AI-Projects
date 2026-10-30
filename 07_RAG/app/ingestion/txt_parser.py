from app.ingestion.base import BaseDocumentParser


class TXTParser(BaseDocumentParser):

    def parse(self, file_path: str) -> list:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        if not text.strip():
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "source": file_path,
                    "file_type": "txt",
                },
            }
        ]