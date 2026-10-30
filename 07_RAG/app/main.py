from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.questions import router as questions_router


app = FastAPI(
    title="RAG Document API",
    version="1.0.0",
)

app.include_router(documents_router)
app.include_router(questions_router)


@app.get("/")
def root():
    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }