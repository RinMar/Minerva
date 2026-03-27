
from sqlalchemy import create_engine, inspect
from src.memory.db import init_db


def test_init_db_applies_migrations(tmp_path, monkeypatch):
    """
    Test that init_db correctly creates the database and all tables
    using Alembic migrations in a temporary directory.
    """
    # Create a temporary database path
    temp_db = tmp_path / "test_memory.db"
    temp_db_uri = f"sqlite:///{temp_db}"

    # Mock DB_PATH in src.paths so Alembic and our code use the temp one
    monkeypatch.setattr("src.memory.db.DATABASE_URL", temp_db_uri)
    monkeypatch.setattr("src.paths.DB_PATH", str(temp_db))

    # Also need to mock the engine in src.memory.db because it's created at import time
    test_engine = create_engine(temp_db_uri)
    monkeypatch.setattr("src.memory.db._engine", test_engine)

    # 1. Run init_db which should trigger Alembic
    init_db()

    # 2. Verify the database file exists
    assert temp_db.exists()

    # 3. Verify all expected tables exist
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()

    # Check for our models' tables
    assert "entities" in tables
    assert "edges" in tables
    assert "embeddings" in tables
    # Check for Alembic's version table
    assert "alembic_version" in tables


def test_init_db_with_existing_tables(tmp_path, monkeypatch):
    """
    Test that init_db correctly handles a situation where tables already exist
    but the alembic_version table does not.
    """
    temp_db = tmp_path / "existing_memory.db"
    temp_db_uri = f"sqlite:///{temp_db}"

    # Mock DB_PATH
    monkeypatch.setattr("src.memory.db.DATABASE_URL", temp_db_uri)
    monkeypatch.setattr("src.paths.DB_PATH", str(temp_db))

    # Pre-create the tables using the old metadata approach
    test_engine = create_engine(temp_db_uri)
    monkeypatch.setattr("src.memory.db._engine", test_engine)
    from src.memory.db import Base
    Base.metadata.create_all(test_engine)

    # 1. Run init_db which should trigger Alembic
    # It should detect tables exist, skip creation, and just stamp the version
    init_db()

    # 2. Verify all expected tables exist
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()

    assert "entities" in tables
    assert "edges" in tables
    assert "embeddings" in tables
    assert "alembic_version" in tables
