from app.llm.base import LLMClient


class MockLLM(LLMClient):
    """Deterministic LLM implementation for testing."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.call_count += 1
        return self.response