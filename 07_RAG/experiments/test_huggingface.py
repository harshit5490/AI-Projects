from app.llm.huggingface import HuggingFaceClient


def main() -> None:

    client = HuggingFaceClient()

    response = client.generate(
        "Explain what FastAPI is in one sentence."
    )

    print("\nGenerated response:")
    print(response)


if __name__ == "__main__":
    main()