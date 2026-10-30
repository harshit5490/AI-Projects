from app.ingestion.docx_parser import DOCXParser


def test_docx_parser():
    parser = DOCXParser()

    documents = parser.parse("data/sample.docx")

    assert len(documents) == 1

    document = documents[0]

    assert "Machine Learning" in document["text"]
    assert "Deep Learning" in document["text"]

    assert document["metadata"]["file_type"] == "docx"
    assert document["metadata"]["source"] == "data/sample.docx"