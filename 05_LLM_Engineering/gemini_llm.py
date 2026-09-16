import os

from dotenv import load_dotenv
from google import genai

from llm_client import LLMClient

load_dotenv()

class GeminiLLM(LLMClient):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.client = genai.client(api_key=api_key)

    def generate(self,prompt:str):
        respone = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )   
        return respone.text 