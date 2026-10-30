import pytest

from app.ingestion.docx_parser import DOCXParser
from app.ingestion.factory import DocumentParserFactory
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.txt_parser import TXTParser


def test_pdf_parser_factory():
    parser = DocumentParserFactory.get_parser("test.pdf")

    assert isinstance(parser, PDFParser)


def test_txt_parser_factory():
    parser = DocumentParserFactory.get_parser("test.txt")

    assert isinstance(parser, TXTParser)


def test_docx_parser_factory():
    parser = DocumentParserFactory.get_parser("test.docx")

    assert isinstance(parser, DOCXParser)


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        DocumentParserFactory.get_parser("test.jpg")