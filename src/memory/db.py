from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from contextlib import contextmanager

os.makedirs("local_store", exist_ok=True)
DATABASE_URL = "sqlite:///local_store/memory.db"

_engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
Base = declarative_base()


@contextmanager
def get_session():
    """Returns an active SQLalchemy session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Import all models and create their tables."""
    from src.memory.schema import EntityNode, GraphEdge, EmbeddingIndex  # noqa: F401
    Base.metadata.create_all(_engine)
