"""
Database setup and connection manager.
Used to provide the SQLAlchemy engine and session access for the local SQLite database,
as well as defining the ORM schema models (EntityNode, GraphEdge, EmbeddingIndex).
"""
from sqlalchemy import create_engine, Column, String, Text, ForeignKey, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager

from src.paths import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

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
    """Import all models and run Alembic migrations to the latest version."""
    from alembic.config import Config
    from alembic import command
    import os

    # Dynamically resolve path to alembic.ini relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    alembic_ini_path = os.path.join(current_dir, "alembic.ini")

    alembic_cfg = Config(alembic_ini_path)
    # Ensure script_location is also absolute to be safe
    alembic_cfg.set_main_option("script_location", os.path.join(current_dir, "alembic"))

    # Run the upgrade to 'head'
    command.upgrade(alembic_cfg, "head")
