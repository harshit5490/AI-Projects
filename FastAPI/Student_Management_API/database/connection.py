from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base

DATABASE_URL = (
    "postgresql://postgres:1234@localhost:5432/student_db"
)

engine = create_engine(
    DATABASE_URL
)

Base = declarative_base()

try:
    with engine.connect() as connection:
        print("Database connected successfully")
except Exception as error:
    print(error)    

