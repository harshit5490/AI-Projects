from app.ingestion.txt_parser import TXTParser


def test_txt_parser():
    parser = TXTParser()

    documents = parser.parse("data/sample.txt")

    assert len(documents) == 1

    document = documents[0]

    assert "Python is a high-level programming language." in document["text"]

    assert document["metadata"]["file_type"] == "txt"

    assert document["metadata"]["source"] == "data/sample.txt"