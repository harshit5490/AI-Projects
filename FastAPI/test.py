from fastapi import FastAPI
from student import router

app = FastAPI()

app.include_router(
    router,
    prefix="/students",
)