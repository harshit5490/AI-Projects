import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


response = client.responses.create(
    model="gpt-5.4",
    input="Explain what a Transformer is in simple language."
)

print(response.output_text)