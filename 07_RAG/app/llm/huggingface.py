import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


class HuggingFaceClient:

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
    ):
        self.model = model

        api_key = os.getenv("HF_TOKEN")

        if not api_key:
            raise ValueError(
                "HF_TOKEN is not configured."
            )

        self.client = InferenceClient(
            api_key=api_key,
            provider="auto",
        )

    def generate(
        self,
        prompt: str,
    ) -> str:

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=256,
            temperature=0.0,
        )

        if not response.choices:
            raise ValueError(
                "Hugging Face returned no choices."
            )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Hugging Face returned an empty response."
            )

        return content.strip()