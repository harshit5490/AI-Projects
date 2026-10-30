from app.llm.huggingface import HuggingFaceClient
from app.retrieval.query_transformer import QueryTransformer


def main() -> None:

    llm_client = HuggingFaceClient()

    transformer = QueryTransformer(
        llm_client=llm_client
    )

    queries = [
        "what's that api thing?",
        "what stores those vectors?",
        "how does that retrieval thing work?",
    ]

    for query in queries:

        print("\n" + "=" * 70)

        print(
            f"Original query:\n{query}"
        )

        rewritten_query = (
            transformer.transform(query)
        )

        print(
            f"\nRewritten query:\n"
            f"{rewritten_query}"
        )


if __name__ == "__main__":
    main()