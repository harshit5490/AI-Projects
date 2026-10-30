
"""
Targeted Q4 answer-quality evaluation.

Question:
    What is semantic search based on?

Uses:
    - Pinecone namespace: rag-evaluation
    - Similarity threshold: 0.35
    - Top-k retrieval: 3
    - One AnswerGenerationService invocation

Note:
    GeminiClient may retry failed API requests according to its
    configured retry policy.
"""

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------
# 1. Project paths and environment
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Override the namespace for this process only.
# Your .env file is not modified.
os.environ["PINECONE_NAMESPACE"] = "rag-evaluation"


# Import after setting the evaluation namespace so that
# environment-based configuration sees the correct value.
from app.dependencies import get_answer_generation_service


# ---------------------------------------------------------
# 2. Evaluation configuration
# ---------------------------------------------------------

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_questions.json"
)

QUESTION_ID = "q4"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.35

ABSTENTION_MESSAGE = (
    "I couldn't find relevant information in the uploaded documents"
)


# ---------------------------------------------------------
# 3. Load only Q4
# ---------------------------------------------------------

def load_q4() -> dict:
    """Load exactly one evaluation question with ID q4."""

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "Expected evaluation_questions.json to contain a JSON list."
        )

    matches = [
        item
        for item in questions
        if item.get("id") == QUESTION_ID
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one question with ID {QUESTION_ID!r}; "
            f"found {len(matches)}."
        )

    item = matches[0]

    if not item.get("question", "").strip():
        raise ValueError("Q4 has an empty question.")

    return item


# ---------------------------------------------------------
# 4. Evaluate answer quality
# ---------------------------------------------------------

def main() -> None:
    question_data = load_q4()

    question = question_data["question"].strip()
    expected_phrases = question_data.get(
        "expected_phrases",
        question_data.get("expected_answer_phrases", []),
    )

    print("=" * 65)
    print("TARGETED RAG ANSWER-QUALITY EVALUATION")
    print("=" * 65)
    print(f"Question ID:          {QUESTION_ID}")
    print(f"Question:             {question}")
    print(f"Pinecone namespace:   rag-evaluation")
    print(f"Top-k:                {TOP_K}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Expected phrases:     {expected_phrases}")
    print()

    # Obtain the configured application service.
    # This call does not itself generate a Gemini answer.
    answer_service = get_answer_generation_service()

    print("Running Q4 through AnswerGenerationService...")
    print("A successful generation should require one Gemini call.")
    print()

    # This is the single answer-generation invocation in this script.
    result = answer_service.answer_question(
        question=question,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    answer = result["answer"]
    sources = result.get("sources", [])

    # -----------------------------------------------------
    # 5. Display the answer
    # -----------------------------------------------------

    print("-" * 65)
    print("GENERATED ANSWER")
    print("-" * 65)
    print(answer)
    print()

    # -----------------------------------------------------
    # 6. Display source attribution
    # -----------------------------------------------------

    print("-" * 65)
    print("RETRIEVED SOURCES")
    print("-" * 65)

    if not sources:
        print("No sources returned.")
    else:
        for index, source in enumerate(sources, start=1):
            print(f"\nSource {index}")
            print(f"  Filename:    {source.get('filename', 'Unknown')}")
            print(f"  Document ID: {source.get('document_id', 'Unknown')}")
            print(f"  File type:   {source.get('file_type', 'Unknown')}")
            print(f"  Chunk index: {source.get('chunk_index', 'Unknown')}")
            print(f"  Page:        {source.get('page')}")
            print(f"  Score:       {source.get('score', 'Unknown')}")
            print(f"  Text:        {source.get('text', 'Unavailable')}")

    print()

    # -----------------------------------------------------
    # 7. Evaluate expected phrases and answer quality
    # -----------------------------------------------------

    normalized_answer = answer.casefold()

    matched_phrases = [
        phrase
        for phrase in expected_phrases
        if phrase.casefold() in normalized_answer
    ]

    missing_phrases = [
        phrase
        for phrase in expected_phrases
        if phrase.casefold() not in normalized_answer
    ]

    abstained = ABSTENTION_MESSAGE.casefold() in normalized_answer

    phrase_coverage = (
        len(matched_phrases) / len(expected_phrases)
        if expected_phrases
        else None
    )

    # Q4 is answerable. A successful result must answer rather
    # than abstain and must contain every configured expected phrase.
    answer_quality_pass = (
        not abstained
        and bool(expected_phrases)
        and not missing_phrases
    )

    source_available = bool(sources)

    print("-" * 65)
    print("EVALUATION RESULTS")
    print("-" * 65)
    print(f"Abstained:          {abstained}")
    print(f"Matched phrases:    {matched_phrases}")
    print(f"Missing phrases:    {missing_phrases}")

    if phrase_coverage is not None:
        print(f"Phrase coverage:    {phrase_coverage:.1%}")
    else:
        print("Phrase coverage:    Not evaluated (no expected phrases configured)")

    print(f"Source available:   {source_available}")
    print(
        "Answer quality:    "
        + ("PASS" if answer_quality_pass else "FAIL")
    )
    print()

    if answer_quality_pass and source_available:
        print(
            "OVERALL: PASS — expected phrases were found "
            "and supporting source references were returned."
        )
    elif answer_quality_pass:
        print(
            "OVERALL: PARTIAL — answer criteria passed, "
            "but no source references were returned."
        )
    else:
        print(
            "OVERALL: FAIL — inspect the answer, retrieved context, "
            "and prompt before changing the pipeline."
        )

    print()
    print(
        "Note: phrase matching is a simple diagnostic, not a complete "
        "measure of factual correctness or grounding."
    )


if __name__ == "__main__":
    main()
