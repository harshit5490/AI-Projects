from pathlib import Path

from app.ingestion.base import BaseDocumentParser
from app.ingestion.docx_parser import DOCXParser
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.txt_parser import TXTParser


class DocumentParserFactory:

    _parsers = {
        ".pdf": PDFParser,
        ".txt": TXTParser,
        ".docx": DOCXParser,
    }

    @classmethod
    def get_parser(cls, file_path: str) -> BaseDocumentParser:
        extension = Path(file_path).suffix.lower()

        parser_class = cls._parsers.get(extension)

        if parser_class is None:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return parser_class()