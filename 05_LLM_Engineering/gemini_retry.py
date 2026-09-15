import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30_000  # 30 seconds
    )
)


# ============================================================
# 3. CHECK WHETHER ERROR IS RETRYABLE
# ============================================================

def is_retryable_error(error: Exception) -> bool:

    # Server errors such as 500 / 503
    if isinstance(error, errors.ServerError):
        return True

    # Client errors
    if isinstance(error, errors.ClientError):

        # 429 = Too Many Requests / Rate Limit
        if getattr(error, "status_code", None) == 429:
            return True

    return False


# ============================================================
# 4. GEMINI REQUEST WITH RETRY
# ============================================================

def generate_with_retry(
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str:

    for attempt in range(max_retries):

        try:

            print(
                f"\nAttempt {attempt + 1}/{max_retries}"
            )

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            return response.text

        except Exception as error:

            # ------------------------------------------------
            # Check if error can be retried
            # ------------------------------------------------

            if not is_retryable_error(error):

                print("\n❌ Non-retryable error")
                print(error)

                raise

            # ------------------------------------------------
            # Last attempt
            # ------------------------------------------------

            if attempt == max_retries - 1:

                print("\n❌ Maximum retries reached")
                print(error)

                raise

            # ------------------------------------------------
            # Exponential Backoff
            # ------------------------------------------------

            exponential_delay = base_delay * (2 ** attempt)

            # Random value between 0 and 1 second
            jitter = random.uniform(0, 1)

            delay = exponential_delay + jitter

            print("\n⚠️ Retryable error")
            print(error)

            print(
                f"Retrying in {delay:.2f} seconds..."
            )

            time.sleep(delay)


# ============================================================
# 5. PROMPT
# ============================================================

prompt = """
You are an AI Engineer mentor.

Explain self-attention in Transformer architecture.

Requirements:

1. Explain Query, Key and Value.
2. Explain how attention scores are calculated conceptually.
3. Give a simple real-world analogy.
4. Keep the answer under 300 words.
"""


# ============================================================
# 6. CALL GEMINI
# ============================================================

try:

    result = generate_with_retry(
        prompt=prompt,
        max_retries=3,
        base_delay=1.0
    )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(result)


except Exception as error:

    print("\n" + "=" * 60)
    print("FINAL FAILURE")
    print("=" * 60)

    print(error)