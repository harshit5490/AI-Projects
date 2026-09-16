import pytest

from app.llm.base import LLMClient


class TestLLMClient(LLMClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "test response"


def test_llm_client_interface():
    client = TestLLMClient()

    response = client.generate(
        system_prompt="You are a test assistant.",
        user_prompt="Hello",
    )

    assert response == "test response"


def test_llm_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMClient()