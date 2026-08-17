from database.connection import Base,engine
from models.student import Student

Base.metadata.create_all(
    bind = engine
)