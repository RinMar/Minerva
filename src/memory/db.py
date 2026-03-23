"""
Database setup and connection manager.
Used to provide the SQLAlchemy engine and session access for the local SQLite database,
as well as defining the ORM schema models (EntityNode, GraphEdge, EmbeddingIndex).
"""
from sqlalchemy import create_engine, Column, String, Text, ForeignKey, Integer
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


class EntityNode(Base):
    __tablename__ = "entities"

    id = Column(String(64), primary_key=True, index=True)
    user_name = Column(String, index=True)
    name = Column(String, index=True)
    topic = Column(String, index=True)
    text = Column(String, nullable=False)
    tags_json = Column(Text)  # Serialized list of strings


class GraphEdge(Base):
    __tablename__ = "edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String, index=True)
    source_id = Column(String(64), ForeignKey("entities.id", ondelete="CASCADE"))
    target_id = Column(String(64), ForeignKey("entities.id", ondelete="CASCADE"))
    source_name = Column(String, index=True)
    target_name = Column(String, index=True)
    relation = Column(String(255))


class EmbeddingIndex(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String, index=True)
    collection = Column(String, index=True)  # "entity" or "edge"
    source_id = Column(String, index=True)   # ID of EntityNode or GraphEdge
    text_content = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)  # Serialized list of floats


def init_db():
    """Import all models and create their tables."""
    Base.metadata.create_all(_engine)
