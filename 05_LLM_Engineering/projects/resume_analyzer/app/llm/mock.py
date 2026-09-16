from app.llm.base import LLMClient


class MockLLM(LLMClient):
    """Deterministic LLM implementation for testing."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return self.response