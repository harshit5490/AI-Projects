
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.llm.gemini import GeminiClient
from app.retrieval.retriever import Retriever
from app.evaluation.abstention_policy import should_abstain


class AnswerGenerationService:
    """Coordinates retrieval, context building, prompting, and generation."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_client: GeminiClient,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        similarity_threshold: float = 0.30,
        metadata_filter: dict | None = None,
    ) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )

        # Abstain if retrieval provides no sufficiently similar chunks.
        if should_abstain(chunks, min_similarity=similarity_threshold):
            return {
                "question": question.strip(),
                "answer": (
                    "I couldn't find relevant information in the uploaded documents to answer this question."
                ),
                "sources": [],
            }

        context = self.context_builder.build(chunks)
        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )
        answer = self.llm_client.generate(prompt)

        sources = []
        seen = set()

        for chunk in chunks:
            source_key = (chunk["document_id"], chunk["chunk_index"])

            if source_key in seen:
                continue

            seen.add(source_key)
            sources.append({
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "file_type": chunk["file_type"],
                "chunk_index": chunk["chunk_index"],
                "page": chunk.get("page"),
                "score": chunk["score"],
                "text": chunk["text"]
            })

        return {
            "question": question.strip(),
            "answer": answer,
            "sources": sources,
        }    