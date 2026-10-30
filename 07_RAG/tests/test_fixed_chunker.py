import pytest

from app.chunking.fixed import FixedSizeChunker


def test_fixed_size_chunking():
    chunker = FixedSizeChunker(chunk_size=10)

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker.split(text)

    assert chunks == [
        "abcdefghij",
        "klmnopqrst",
        "uvwxyz",
    ]


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0)