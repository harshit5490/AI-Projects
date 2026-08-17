from fastapi import FastAPI

from router.student import router


app = FastAPI()


app.include_router(
    router,
    prefix="/students",
    tags=["Students"]
)