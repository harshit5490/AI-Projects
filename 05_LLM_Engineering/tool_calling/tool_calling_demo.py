import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import (
    add_numbers,
    multiply_numbers,
    get_user_info,
)


# ============================================================
# 1. LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. DEFINE AVAILABLE TOOLS
# ============================================================

tools = [
    add_numbers,
    multiply_numbers,
    get_user_info,
]


# ============================================================
# 4. CREATE MODEL CONFIGURATION
# ============================================================

config = types.GenerateContentConfig(
    tools=tools
)


# ============================================================
# 5. SEND USER REQUEST
# ============================================================

user_prompt = "What is 25 multiplied by 8?"

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=user_prompt,
    config=config
)


# ============================================================
# 6. INSPECT RESPONSE
# ============================================================

print("=" * 60)
print("GEMINI RESPONSE")
print("=" * 60)

print(response.text)