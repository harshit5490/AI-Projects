import os

import httpx
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN is not configured.")

    response = httpx.get(
        "https://router.huggingface.co/v1/models",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    for model in data["data"]:
        print(model["id"])


if __name__ == "__main__":
    main()