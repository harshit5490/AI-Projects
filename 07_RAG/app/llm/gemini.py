
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiClient:
    """Generates text using Google's Gemini API."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL")

        if not self.model:
            raise ValueError(
                "GEMINI_MODEL is not configured."
            )

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Generate a response for the supplied prompt."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": 512,
            },
        )

        if not response.text or not response.text.strip():
            raise ValueError(
                "Gemini returned an empty response."
            )

        return response.text.strip()
