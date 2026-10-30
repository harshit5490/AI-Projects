
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.embeddings.service import EmbeddingService
from app.vectorstore.store import PineconeVectorStore
from app.retrieval.retriever import Retriever
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.llm.gemini import GeminiClient
from app.services.answer_generation_service import AnswerGenerationService
from app.evaluation.abstention_metrics import abstention_metrics


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "evaluation_questions.json"
)

THRESHOLD = 0.35
TOP_K = 3


def main() -> None:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    embedding_service = EmbeddingService()

    vector_store = PineconeVectorStore(
        index_name=os.getenv(
            "PINECONE_INDEX_NAME",
            "rag-document-index",
        ),
        namespace="rag-evaluation",
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    answer_service = AnswerGenerationService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_client=GeminiClient(),
    )

    answerable_labels = []
    abstention_decisions = []

    total_expected_phrases = 0
    matched_expected_phrases = 0
    answerable_phrase_passes = 0
    answerable_count = 0
    source_available_count = 0

    print(
        f"Answer Quality Evaluation | "
        f"Threshold={THRESHOLD} | Top-K={TOP_K}"
    )
    print("=" * 72)

    for item in questions:
        question = item["question"]
        expected_answerable = item["answerable"]

        result = answer_service.answer_question(
            question=question,
            top_k=TOP_K,
            similarity_threshold=THRESHOLD,
        )

        answer = result["answer"]
        sources = result["sources"]
        abstained = len(sources) == 0

        answerable_labels.append(expected_answerable)
        abstention_decisions.append(abstained)

        print(f"\n[{item['id']}] {question}")
        print(f"Expected answerable: {expected_answerable}")
        print(f"Abstained: {abstained}")
        print(f"Sources returned: {len(sources)}")
        print(f"Answer: {answer}")

        if sources:
            source_available_count += 1

        if expected_answerable:
            answerable_count += 1

            normalized_answer = answer.casefold()
            phrases = item["expected_phrases"]

            matched = [
                phrase
                for phrase in phrases
                if phrase.casefold() in normalized_answer
            ]

            total_expected_phrases += len(phrases)
            matched_expected_phrases += len(matched)

            phrase_pass = (
                bool(phrases)
                and len(matched) == len(phrases)
                and not abstained
            )

            if phrase_pass:
                answerable_phrase_passes += 1

            print(f"Expected phrases: {phrases}")
            print(f"Matched phrases: {matched}")
            print(f"Phrase check passed: {phrase_pass}")

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)

    metrics = abstention_metrics(
        answerable_labels,
        abstention_decisions,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    phrase_coverage = (
        matched_expected_phrases / total_expected_phrases
        if total_expected_phrases
        else 0.0
    )

    answer_pass_rate = (
        answerable_phrase_passes / answerable_count
        if answerable_count
        else 0.0
    )

    print(
        f"expected_phrase_coverage: {phrase_coverage:.3f}"
    )
    print(
        f"answerable_phrase_pass_rate: {answer_pass_rate:.3f}"
    )
    print(
        f"source_availability_rate: "
        f"{source_available_count / len(questions):.3f}"
    )


if __name__ == "__main__":
    main()
