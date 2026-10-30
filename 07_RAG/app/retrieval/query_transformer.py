from app.llm.huggingface import HuggingFaceClient


class QueryTransformer:

    def __init__(
        self,
        llm_client: HuggingFaceClient,
    ):
        self.llm_client = llm_client

    def transform(
        self,
        query: str,
    ) -> str:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        prompt = f"""
Rewrite the following user query into a clear,
concise search query for semantic retrieval.

Rules:
- Preserve the original meaning.
- Preserve important constraints.
- Do not answer the question.
- Do not add information that is not present.
- Return only the rewritten query.

Original query:
{query}
"""

        rewritten_query = (
            self.llm_client.generate(
                prompt
            )
        )

        rewritten_query = (
            rewritten_query.strip()
        )

        if not rewritten_query:
            raise ValueError(
                "Query transformation returned an empty query."
            )

        return rewritten_query