
import pytest

from app.llm.gemini import GeminiClient


def test_client_requires_model(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="GEMINI_MODEL is not configured",
    ):
        GeminiClient()


def test_client_requires_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY is not configured",
    ):
        GeminiClient()


def test_generate_rejects_empty_prompt(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    client = GeminiClient()

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        client.generate("   ")
