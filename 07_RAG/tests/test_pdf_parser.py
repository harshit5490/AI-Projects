from app.ingestion.pdf_parser import PDFParser


def test_pdf_parser():
    parser = PDFParser()

    documents = parser.parse("data/sample.pdf")

    assert len(documents) > 0

    first_document = documents[0]

    assert "text" in first_document
    assert "metadata" in first_document

    assert first_document["metadata"]["file_type"] == "pdf"
    assert first_document["metadata"]["page"] == 1