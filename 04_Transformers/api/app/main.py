from fastapi import FastAPI
from pydantic import BaseModel

from app.qa_model import (
    predict_answer,
    MODEL_NAME
)


# ============================================================
# 1. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="QA System API",
    description="Extractive Question Answering API using RoBERTa SQuAD2",
    version="1.0.0"
)


# ============================================================
# 2. REQUEST AND RESPONSE DATA MODEL
# ============================================================

class QARequest(BaseModel):

    question: str
    context: str


class QAResponse(BaseModel):

    question: str
    answer: str
    has_answer: bool
    best_span_score: float
    null_score: float
    score_difference: float


# ============================================================
# 3. HOME AND HEALTH ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "QA System API is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


# ============================================================
# 4. PREDICT ENDPOINT
# ============================================================

@app.post("/predict", response_model=QAResponse)
def predict(request: QARequest):

    result = predict_answer(
        request.question,
        request.context
    )

    return {
        "question": request.question,
        **result
    }